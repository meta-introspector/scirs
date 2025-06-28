# Unwrap() Usage Report

Total unwrap() calls and unsafe operations found: 3905

## Summary by Type

- Division without zero check - use safe_divide(): 2292 occurrences
- Replace with ? operator or .ok_or(): 1186 occurrences
- Mathematical operation .sqrt() without validation: 252 occurrences
- Mathematical operation .ln() without validation: 86 occurrences
- Mathematical operation .powf( without validation: 48 occurrences
- Handle array creation errors properly: 21 occurrences
- Use .get() with proper bounds checking: 11 occurrences
- Use .get().ok_or(Error::IndexOutOfBounds)?: 9 occurrences

## Detailed Findings


### benches/wavelet_bench.rs

17 issues found:

- Line 11: `let t = (0..size).map(|i| i as f64 / fs).collect::<Vec<f64>>();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 13: `chirp(&t, 0.0, 1.0, 100.0, "linear", 0.5).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 40: `|b, _| b.iter(|| black_box(dwt_decompose(&signal, *wavelet, None).unwrap())),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 45: `let (approx, detail) = dwt_decompose(&signal, *wavelet, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 51: `|b, _| b.iter(|| black_box(dwt_reconstruct(&approx, &detail, *wavelet).unwrap())...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 93: `b.iter(|| black_box(wavedec(&signal, *wavelet, Some(level), None).unwrap()))`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 98: `let coeffs = wavedec(&signal, *wavelet, Some(level), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 104: `|b, _| b.iter(|| black_box(waverec(&coeffs, *wavelet).unwrap())),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 146: `b.iter(|| black_box(swt_decompose(&signal, *wavelet, level, None).unwrap()))`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 151: `let (approx, detail) = swt_decompose(&signal, *wavelet, level, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 159: `black_box(swt_reconstruct(&approx, &detail, *wavelet, level).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 170: `|b, _| b.iter(|| black_box(swt(&signal, *wavelet, level, None).unwrap())),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 174: `let (details, approx) = swt(&signal, *wavelet, level, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 180: `|b, _| b.iter(|| black_box(iswt(&details, &approx, *wavelet).unwrap())),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 223: `b.iter(|| black_box(wp_decompose(&signal, *wavelet, level, None).unwrap()))`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 228: `let tree = wp_decompose(&signal, *wavelet, level, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 240: `|b, _| b.iter(|| black_box(reconstruct_from_nodes(&tree, &nodes).unwrap())),`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/adaptive_filter_example.rs

20 issues found:

- Line 25: `let mut lms = LmsFilter::new(4, 0.01, 0.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 51: `let (_outputs, errors, _mse) = lms.adapt_batch(&inputs, &desired).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 63: `.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 67: `let final_mse: f64 = errors.iter().rev().take(50).map(|&e| e * e).sum::<f64>() /...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 97: `let mut nlms = NlmsFilter::new(8, 0.5, 1e-6).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 106: `let (noise_est, _error, _mse) = nlms.adapt(noise_signal[i], noisy_signal[i]).unw...`
  - **Fix**: Use .get() with proper bounds checking
- Line 133: `let mut lms_comp = LmsFilter::new(3, 0.05, 0.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 134: `let mut rls_comp = RlsFilter::new(3, 0.99, 100.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 155: `let (_out_lms, err_lms, _) = lms_comp.adapt(input, desired).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 156: `let (_out_rls, err_rls, _) = rls_comp.adapt(input, desired).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 174: `.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 182: `.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 204: `let mut echo_canceler = LmsFilter::new(8, 0.01, 0.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 247: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 255: `echo_signal.iter().map(|&x| x * x).sum::<f64>() / echo_signal.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 262: `/ 50.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 263: `let echo_cancellation_db = 10.0 * (echo_power / residual_power).log10();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 297: `let signal_power: f64 = signal.iter().map(|&x| x * x).sum::<f64>() / signal.len(...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 298: `let noise_power: f64 = noise.iter().map(|&x| x * x).sum::<f64>() / noise.len() a...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 301: `10.0 * (signal_power / noise_power).log10()`
  - **Fix**: Division without zero check - use safe_divide()

### examples/advanced_filter_example.rs

27 issues found:

- Line 25: `let (b_butter, a_butter) = butter(4, 0.3, "lowpass").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 30: `let (b_butter_hp, a_butter_hp) = butter(3, 0.4, "highpass").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 39: `let (b_cheb1, a_cheb1) = cheby1(4, 0.5, 0.3, "lowpass").unwrap(); // 0.5 dB ripp...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 48: `let (b_cheb2, a_cheb2) = cheby2(4, 40.0, 0.3, "lowpass").unwrap(); // 40 dB stop...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 57: `let (b_ellip, a_ellip) = ellip(4, 0.5, 40.0, 0.3, "lowpass").unwrap(); // 0.5 dB...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 66: `let (b_bessel, a_bessel) = bessel(4, 0.3, "lowpass").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 73: `let gd_butter = group_delay(&b_butter, &a_butter, &frequencies).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 74: `let gd_bessel = group_delay(&b_bessel, &a_bessel, &frequencies).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 91: `let (b_comb_fir, _a_comb_fir) = comb_filter(20, 0.0, 0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 100: `let (_b_comb_iir, a_comb_iir) = comb_filter(15, -0.7, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 113: `let (b_notch, a_notch) = notch_filter(0.12, 30.0).unwrap(); // 60/(1000/2) = 0.1...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 119: `let (b_peak, a_peak) = peak_filter(0.2, 5.0, 6.0).unwrap(); // 6 dB boost at 0.2...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 129: `let (b_ap1, a_ap1) = allpass_filter(0.25, 0.9).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 135: `let (b_ap2, a_ap2) = allpass_filter(0.3, 0.8).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 146: `let t: Vec<f64> = (0..1000).map(|i| i as f64 / fs).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 159: `let filtered_butter = lfilter(&b_butter, &a_butter, &signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 160: `let filtered_cheb1 = lfilter(&b_cheb1, &a_cheb1, &signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 161: `let filtered_ellip = lfilter(&b_ellip, &a_ellip, &signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 162: `let filtered_bessel = lfilter(&b_bessel, &a_bessel, &signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 163: `let filtered_notch = lfilter(&b_notch, &a_notch, &signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 178: `100.0 * rms_butter / rms_original`
  - **Fix**: Division without zero check - use safe_divide()
- Line 183: `100.0 * rms_cheb1 / rms_original`
  - **Fix**: Division without zero check - use safe_divide()
- Line 188: `100.0 * rms_ellip / rms_original`
  - **Fix**: Division without zero check - use safe_divide()
- Line 193: `100.0 * rms_bessel / rms_original`
  - **Fix**: Division without zero check - use safe_divide()
- Line 198: `100.0 * rms_notch / rms_original`
  - **Fix**: Division without zero check - use safe_divide()
- Line 233: `(sum_squares / signal.len() as f64).sqrt()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 233: `(sum_squares / signal.len() as f64).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation

### examples/advanced_processing_example.rs

15 issues found:

- Line 35: `let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 58: `let window = hann(window_length, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 69: `MemoryEfficientStft::new(&window, hop_size, fs, Some(stft_config), memory_config...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 77: `let spectrogram = mem_stft.spectrogram_chunked(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 98: `let hann_win = hann(length, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 99: `let hamming_win = window::hamming(length, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 100: `let blackman_win = window::blackman(length, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 101: `let kaiser_win = kaiser(length, 8.0, true).unwrap(); // High beta for good sidel...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 113: `let comparison = compare_windows(&windows).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 143: `let designed_window = design_window_with_constraints(64, sidelobe_req, None).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 144: `let analysis = analyze_window(&designed_window, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 165: `let x_fine: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 246: `let freq_step = spectrogram.shape()[0] / 50;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 300: `/ y_true.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 301: `mse.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation

### examples/bss_example.rs

21 issues found:

- Line 49: `sources[[2, i]] = phase / PI - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 68: `let normal = Normal::new(0.0, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 82: `.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 98: `let normal = Normal::new(0.0, noise_level).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 123: `let orig_mean = orig.mean().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 124: `let rec_mean = rec.mean().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 140: `correlations[[i, j]] = numerator / (orig_var.sqrt() * rec_var.sqrt());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 140: `correlations[[i, j]] = numerator / (orig_var.sqrt() * rec_var.sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 175: `total_correlation / n_orig as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 191: `let min_len = signals.iter().map(|(_, data)| data.len()).min().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 330: `sources[[1, i]] = (-x * x / 2.0).exp() + (-((x - 3.0) * (x - 3.0)) / 1.0).exp() ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 336: `sources[[2, i]] = (-x / 2.0).exp() + (-(x - 7.0) / 1.0).exp() * (x >= 7.0) as i3...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 471: `let mean = component.mean().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 472: `let var = component.mapv(|x| (x - mean).powi(2)).sum() / (n_samples as f64 - 1.0...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 477: `variances.sort_by(|a, b| b.partial_cmp(a).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 489: `var / total_var * 100.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 490: `cum_var / total_var * 100.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 547: `if phase < PI / 4.0 || (phase > PI && phase < PI + PI / 4.0) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 556: `sources[[2, i]] = (-x * x / 2.0).exp() * (2.0 * PI * x).cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 626: `let sparsity_orig_pct = sparsity_original as f64 / (n_sources * n_samples) as f6...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 627: `let sparsity_rec_pct = sparsity_recovered as f64 / (n_sources * n_samples) as f6...`
  - **Fix**: Division without zero check - use safe_divide()

### examples/complex_wavelets_example.rs

5 issues found:

- Line 17: `let t: Vec<f64> = (0..num_samples).map(|i| i as f64 / fs).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 31: `let factor = f64::powf(max_scale / min_scale, 1.0 / (num_scales - 1) as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 163: `.map(|&scale_idx| center_freq * fs / (scales[scale_idx] * 2.0 * std::f64::consts...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 175: `.map(|&t_val| 5.0 + (100.0 - 5.0) * t_val / duration)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 187: `/ (freq.len() - 2 * skip) as f64`
  - **Fix**: Division without zero check - use safe_divide()

### examples/constant_q_transform_example.rs

28 issues found:

- Line 72: `let rate = (880.0_f64 / 110.0).ln() / duration;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 72: `let rate = (880.0_f64 / 110.0).ln() / duration;`
  - **Fix**: Mathematical operation .ln() without validation
- Line 73: `let phase = 2.0 * PI * 110.0 * (rate * ti).exp() / rate;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 84: `let segment = (ti / 0.5).floor() as usize % 4;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 130: `let cqt_result = constant_q_transform(signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 170: `let cqt_result = constant_q_transform(signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 187: `peaks.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 204: `let chroma = chromagram(&cqt_result, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 230: `let cqt_result = constant_q_transform(signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 240: `for &t in cqt_result.times.as_ref().unwrap().iter() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 266: `cqt_result.times.as_ref().unwrap()[0],`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 267: `cqt_result.times.as_ref().unwrap()[cqt_result.times.as_ref().unwrap().len() - 1]`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 283: `let cqt_result = constant_q_transform(signal, &cqt_config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 299: `signal.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 306: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 310: `let psd_db: Vec<f64> = psd.iter().map(|&p| 10.0 * (p / max_psd).log10()).collect...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 329: `/ (cqt_result.frequencies.len() - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 334: `(cqt_result.frequencies[1] / cqt_result.frequencies[0])`
  - **Fix**: Division without zero check - use safe_divide()
- Line 370: `let cqt_result = constant_q_transform(signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 373: `let reconstructed = inverse_constant_q_transform(&cqt_result, Some(signal.len())...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 383: `let reconstruction_snr = 10.0 * (signal_power / total_error).log10();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 413: `let cqt_result = constant_q_transform(signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 416: `let chroma = chromagram(&cqt_result, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 423: `for &t in cqt_result.times.as_ref().unwrap().iter() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 444: `cqt_result.times.as_ref().unwrap()[cqt_result.times.as_ref().unwrap().len() - 1]`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 461: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 476: `segment_chroma[pc] += chroma[[pc, frame]] / segment_frames.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 487: `pc_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/cwt_scalogram_example.rs

6 issues found:

- Line 16: `let fs = 1.0 / dt;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 32: `.map(|i| 2.0_f64.powf(i as f64 / 8.0))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 32: `.map(|i| 2.0_f64.powf(i as f64 / 8.0))`
  - **Fix**: Mathematical operation .powf( without validation
- Line 43: `let central_freq = 0.85 / (2.0 * std::f64::consts::PI);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 63: `let time_step = n / 20;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 64: `let freq_step = num_scales / 10;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/deconvolution_example.rs

40 issues found:

- Line 35: `(-((xi - 3.0).powi(2) / 0.5)).exp() + 0.7 * (-((xi - 7.0).powi(2) / 0.3)).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 43: `psf[i] = (-(xi.powi(2) / (2.0 * psf_width.powi(2)))).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 52: `true_signal.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 53: `psf.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 61: `let normal = Normal::new(0.0, noise_level).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 101: `/ n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 108: `/ n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 115: `/ n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 151: `sparse_signal[n / 4] = 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 152: `sparse_signal[n / 2] = 0.7;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 153: `sparse_signal[3 * n / 4] = 0.5;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 160: `0.8 * (-((xi - 3.0).powi(2) / 1.0)).exp() + (-((xi - 7.0).powi(2) / 0.5)).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 168: `psf[i] = (-(xi.powi(2) / (2.0 * psf_width.powi(2)))).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 179: `true_signal.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 180: `psf.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 188: `let normal = Normal::new(0.0, noise_level).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 272: `true_signal[i] = (-((xi - 3.0).powi(2) / 0.5)).exp()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 273: `+ 0.5 * (-((xi - 6.0).powi(2) / 0.3)).exp()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 274: `+ 0.3 * (-((xi - 8.0).powi(2) / 0.1)).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 285: `psf[i] = (-(xi.powi(2) / (2.0f64 * width.powi(2)))).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 295: `psf[n / 2 - blur_length / 2 + i] = 1.0 / blur_length as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 304: `let xi = x[i] - x[n / 2];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 319: `true_signal.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 320: `psf.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 328: `let normal = Normal::new(0.0, noise_level).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 392: `(-((xi - 3.0f64).powi(2) / 0.5)).exp() + 0.7 * (-((xi - 7.0f64).powi(2) / 0.3))....`
  - **Fix**: Division without zero check - use safe_divide()
- Line 400: `true_psf[i] = (-(xi.powi(2) / (2.0f64 * true_psf_width.powi(2)))).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 406: `true_signal.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 407: `true_psf.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 415: `let normal = Normal::new(0.0, noise_level).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 486: `true_signal[i] = 2.0 * (-((xi - 2.0).powi(2) / 0.2)).exp()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 487: `+ 1.5 * (-((xi - 5.0).powi(2) / 0.3)).exp()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 488: `+ 1.0 * (-((xi - 8.0).powi(2) / 0.1)).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 496: `psf[i] = (-(xi.powi(2) / (2.0f64 * psf_width.powi(2)))).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 502: `true_signal.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 503: `psf.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 511: `let normal = Normal::new(0.0, noise_level).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 551: `.min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 553: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 616: `/ estimate.len() as f64`
  - **Fix**: Division without zero check - use safe_divide()

### examples/dmeyer_wavelet_example.rs

6 issues found:

- Line 12: `let signal = chirp(&t, 0.0, 1.0, 100.0, "linear", 0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 54: `let coeffs = wavedec(&noisy_signal, wavelet, Some(3), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 95: `let denoised_signal = waverec(&denoised_coeffs, wavelet).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 103: `decomp_time.as_micros() as f64 / 1000.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 104: `recon_time.as_micros() as f64 / 1000.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 113: `/ signal.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/dpss_example.rs

19 issues found:

- Line 16: `let window = dpss(n, nw, None, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 28: `let concentration = center_energy / total_energy;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 41: `let windows = dpss_windows(n, nw, Some(num_tapers), true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 77: `let window = dpss(n, nw, None, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 84: `let freq = k as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 89: `let effective_bandwidth = weighted_freq_sum / weight_sum;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 104: `("Hann", get_window("hann", n, false).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 105: `("Hamming", get_window("hamming", n, false).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 106: `("DPSS (NW=2.5)", dpss(n, 2.5, None, true).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 107: `("DPSS (NW=4.0)", dpss(n, 4.0, None, true).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 118: `let center = n / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 131: `let exclude_range = main_lobe_width / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 151: `let window = dpss(n, 3.0, None, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 163: `let phase = -2.0 * PI * (k as f64) * (j as f64) / (nfft as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 168: `freq_response[k] = (real_sum * real_sum + imag_sum * imag_sum).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 173: `let threshold_3db = max_response / 2.0_f64.sqrt(); // -3dB`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 177: `for i in 1..nfft / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 189: `bandwidth_3db as f64 / nfft as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 195: `let freq_norm = i as f64 / nfft as f64;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/emd_example.rs

1 issues found:

- Line 25: `let time: Vec<f64> = (0..n_samples).map(|i| i as f64 / sample_rate).collect();`
  - **Fix**: Division without zero check - use safe_divide()

### examples/filter_banks_example.rs

15 issues found:

- Line 21: `let t: Vec<f64> = (0..n_samples).map(|i| i as f64 / fs).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 81: `reconstruction_error = (reconstruction_error / min_len as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 81: `reconstruction_error = (reconstruction_error / min_len as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 149: `wavelet_error = (wavelet_error / wavelet_min_len as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 149: `wavelet_error = (wavelet_error / wavelet_min_len as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 188: `cmfb_error = (cmfb_error / cmfb_min_len as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 188: `cmfb_error = (cmfb_error / cmfb_min_len as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 219: `test_error = (test_error / test_min_len as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 219: `test_error = (test_error / test_min_len as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 239: `println!("- Numerator (b): {:?}", b_unstable.as_slice().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 240: `println!("- Denominator (a): {:?}", a_unstable.as_slice().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 253: `println!("- Stabilized numerator: {:?}", b_stable.as_slice().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 256: `a_stable.as_slice().unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 275: `let freq_resolution = fs / (2.0 * channels as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 276: `let time_resolution = channels as f64 / fs;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/higher_order_spectra_example.rs

14 issues found:

- Line 27: `bispectrum(&signal_phase_coupled, nfft, Some("hann"), None, fs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 37: `bispectrum(&signal_no_coupling, nfft, Some("hann"), None, fs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 51: `bicoherence(&signal_phase_coupled, nfft, Some("hann"), None, fs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 54: `bicoherence(&signal_no_coupling, nfft, Some("hann"), None, fs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 74: `detect_phase_coupling(&signal_phase_coupled, nfft, Some("hann"), fs, Some(0.6))....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 89: `let tri_coupled = trispectrum(&signal_phase_coupled, nfft, Some("hann"), fs).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 96: `let (biamp, (_, _)) = biamplitude(&signal_phase_coupled, nfft, Some("hann"), fs)...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 104: `cumulative_bispectrum(&signal_phase_coupled, nfft, Some("hann"), fs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 112: `skewness_spectrum(&signal_phase_coupled, nfft, Some("hann"), fs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 119: `let angles = [0.0, PI / 4.0, PI / 2.0, PI, 3.0 * PI / 2.0];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 124: `let (bicoh, (_, _)) = bicoherence(&signal, nfft, Some("hann"), None, fs).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 145: `let t = Array1::linspace(0.0, (n_samples as f64 - 1.0) / fs, n_samples);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 179: `let t = Array1::linspace(0.0, (n_samples as f64 - 1.0) / fs, n_samples);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 211: `let t = Array1::linspace(0.0, (n_samples as f64 - 1.0) / fs, n_samples);`
  - **Fix**: Division without zero check - use safe_divide()

### examples/image_feature_analysis.rs

45 issues found:

- Line 51: `image[[i, j]] = (i as f64 / size as f64 + j as f64 / size as f64) * 127.5;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 66: `if (i / 4 + j / 4) % 2 == 0 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 83: `let center_x = size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 84: `let center_y = size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 85: `let radius = size / 4;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 91: `let distance = ((dx * dx + dy * dy) as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 131: `if (i + 1) % (images.len() / 10) == 0 || i + 1 == images.len() {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 136: `(i + 1) as f64 / images.len() as f64 * 100.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 166: `let n_classes = *labels.iter().max().unwrap() + 1;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 193: `let mean = sum / values.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 196: `values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 197: `let stddev = variance.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 226: `stddevs.iter().sum::<f64>() / stddevs.len() as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 233: `max_diff / avg_stddev`
  - **Fix**: Division without zero check - use safe_divide()
- Line 261: `for i in 0..size / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 262: `for j in 0..size / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 263: `image[[i, j]] = (i + j) as f64 / (size as f64) * 255.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 268: `for i in 0..size / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 269: `for j in size / 2..size {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 270: `if (i / 4 + j / 4) % 2 == 0 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 279: `let center_x = size / 4 * 3;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 280: `let center_y = size / 4 * 3;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 281: `let _radius = size / 6;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 283: `for i in size / 2..size {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 284: `for j in 0..size / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 287: `let distance = ((dx * dx + dy * dy) as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 289: `if (distance / 4.0).floor() % 2.0 == 0.0 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 298: `for i in size / 2..size {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 299: `for j in size / 2..size {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 378: `let center = (i + patch_size / 2, j + patch_size / 2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 415: `.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 476: `.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 507: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 533: `for i in 0..size / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 534: `for j in 0..size / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 543: `for i in 0..size / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 544: `for j in size / 2..size {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 553: `for i in size / 2..size {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 554: `for j in 0..size / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 563: `for i in size / 2..size {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 564: `for j in size / 2..size {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 585: `.map(|&count| format!("{:.1}%", count as f64 / total * 100.0))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 634: `let distance = ((dx * dx + dy * dy) as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 638: `let intensity = 180.0 - distance / radius as f64 * 100.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 683: `let distance = ((dx * dx + dy * dy) as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation

### examples/image_feature_extraction.rs

13 issues found:

- Line 23: `let center_x = size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 24: `let center_y = size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 25: `let radius = size / 4;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 30: `let gradient = (i + j) as f64 / (2 * size) as f64 * 255.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 35: `let distance = ((dx * dx + dy * dy) as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 65: `let r_gradient = i as f64 / size as f64 * 255.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 66: `let g_gradient = j as f64 / size as f64 * 255.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 67: `let b_gradient = ((i as f64 / size as f64) * (j as f64 / size as f64)) * 255.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 72: `let distance = ((dx * dx + dy * dy) as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 108: `smooth_texture[[i, j]] = (i as f64 + j as f64) / (2.0 * size as f64) * 255.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 111: `if (i / 4 + j / 4) % 2 == 0 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 130: `let smooth_features = extract_image_features(&smooth_texture, &texture_options)....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 131: `let rough_features = extract_image_features(&rough_texture, &texture_options).un...`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/kalman_filter_example.rs

27 issues found:

- Line 44: `let normal = Normal::new(0.0, noise_level).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 78: `let min_len = signals.iter().map(|(_, data)| data.len()).min().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 142: `/ n_samples as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 149: `/ n_samples as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 156: `/ n_samples as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 199: `Array2::from_shape_vec((2, 2), vec![1.0, dt, -dt * (x[0]).cos(), 1.0 - 0.1 * dt]...`
  - **Fix**: Handle array creation errors properly
- Line 204: `Array2::from_shape_vec((1, 2), vec![2.0 * x[0], 0.0]).unwrap()`
  - **Fix**: Handle array creation errors properly
- Line 226: `let normal = Normal::new(0.0, noise_level).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 259: `/ n_samples as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 266: `/ n_samples as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 328: `let normal = Normal::new(0.0, noise_level).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 367: `/ n_samples as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 374: `/ n_samples as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 412: `let x = (i - 200) as f64 / 100.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 420: `let normal = Normal::new(0.0, noise_level).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 450: `/ n_samples as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 457: `/ n_samples as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 464: `/ n_samples as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 471: `/ n_samples as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 509: `let noise_level = 0.1 + (i as f64 / n_samples as f64) * 0.5;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 510: `let normal = Normal::new(0.0, noise_level).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 534: `/ n_samples as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 541: `/ n_samples as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 548: `/ n_samples as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 608: `/ n_samples as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 615: `/ n_samples as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 622: `/ n_samples as f64;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/lombscargle_example.rs

3 issues found:

- Line 69: `frequencies.first().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 70: `frequencies.last().unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 138: `let max_idx = power.iter().position(|&p| p == max_power).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/lti_interconnection_example.rs

36 issues found:

- Line 16: `let g1 = tf(vec![5.0], vec![1.0, 2.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 17: `println!("G1(s) = 5 / (s + 2)");`
  - **Fix**: Division without zero check - use safe_divide()
- Line 20: `let g2 = tf(vec![1.0], vec![1.0, 5.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 21: `println!("G2(s) = 1 / (s + 5)");`
  - **Fix**: Division without zero check - use safe_divide()
- Line 24: `let series_sys = series(&g1, &g2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 36: `let g1_par = tf(vec![3.0], vec![1.0, 1.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 37: `println!("G1(s) = 3 / (s + 1)");`
  - **Fix**: Division without zero check - use safe_divide()
- Line 40: `let g2_par = tf(vec![2.0], vec![1.0, 4.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 41: `println!("G2(s) = 2 / (s + 4)");`
  - **Fix**: Division without zero check - use safe_divide()
- Line 44: `let parallel_sys = parallel(&g1_par, &g2_par).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 55: `let plant = tf(vec![10.0], vec![1.0, 1.0, 0.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 56: `println!("Plant G(s) = 10 / (s(s + 1))");`
  - **Fix**: Division without zero check - use safe_divide()
- Line 59: `let closed_loop = feedback(&plant, None, 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 60: `println!("Closed-loop T(s) = G(s) / (1 + G(s))");`
  - **Fix**: Division without zero check - use safe_divide()
- Line 76: `let plant_pid = tf(vec![1.0], vec![1.0, 3.0, 2.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 77: `println!("Plant G(s) = 1 / ((s + 1)(s + 2))");`
  - **Fix**: Division without zero check - use safe_divide()
- Line 81: `let controller = tf(vec![2.0, 10.0, 5.0], vec![1.0, 0.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 82: `println!("PID Controller C(s) = (2s^2 + 10s + 5) / s");`
  - **Fix**: Division without zero check - use safe_divide()
- Line 85: `let forward_path = series(&controller, &plant_pid).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 89: `let pid_closed_loop = feedback(&forward_path, None, 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 99: `let sens = sensitivity(&plant, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 100: `let comp_sens = complementary_sensitivity(&plant, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 102: `println!("Sensitivity S(s) = 1 / (1 + G(s)):");`
  - **Fix**: Division without zero check - use safe_divide()
- Line 106: `println!("Complementary Sensitivity T(s) = G(s) / (1 + G(s)):");`
  - **Fix**: Division without zero check - use safe_divide()
- Line 111: `let sum_st = parallel(&sens, &comp_sens).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 129: `let open_loop_resp = plant.frequency_response(&[freq]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 130: `let closed_loop_resp = closed_loop.frequency_response(&[freq]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 147: `let (w, mag, phase) = bode(&closed_loop, Some(&bode_freqs)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 164: `let prefilter = tf(vec![1.0], vec![0.1, 1.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 165: `println!("Prefilter F(s) = 1 / (0.1s + 1)");`
  - **Fix**: Division without zero check - use safe_divide()
- Line 168: `let prop_controller = tf(vec![5.0], vec![1.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 172: `let simple_plant = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 173: `println!("Plant G(s) = 1 / (s + 1)");`
  - **Fix**: Division without zero check - use safe_divide()
- Line 176: `let controller_plant = series(&prop_controller, &simple_plant).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 177: `let closed_inner = feedback(&controller_plant, None, 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 178: `let complete_system = series(&prefilter, &closed_inner).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/median_filter_example.rs

40 issues found:

- Line 36: `let filtered_standard = median_filter_1d(&noisy_signal, 5, &standard_config).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 37: `let filtered_adaptive = median_filter_1d(&noisy_signal, 5, &adaptive_config).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 38: `let filtered_weighted = median_filter_1d(&noisy_signal, 5, &center_weighted_conf...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 50: `100.0 * (mse_noisy - mse_standard) / mse_noisy`
  - **Fix**: Division without zero check - use safe_divide()
- Line 55: `100.0 * (mse_noisy - mse_adaptive) / mse_noisy`
  - **Fix**: Division without zero check - use safe_divide()
- Line 60: `100.0 * (mse_noisy - mse_weighted) / mse_noisy`
  - **Fix**: Division without zero check - use safe_divide()
- Line 78: `let min_filter = rank_filter_1d(&noisy_signal, 5, 0.0, EdgeMode::Reflect).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 79: `let median_filter = rank_filter_1d(&noisy_signal, 5, 0.5, EdgeMode::Reflect).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 80: `let max_filter = rank_filter_1d(&noisy_signal, 5, 1.0, EdgeMode::Reflect).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 97: `let image_standard = median_filter_2d(&noisy_image, 3, &standard_config).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 98: `let image_adaptive = median_filter_2d(&noisy_image, 3, &adaptive_config).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 99: `let image_weighted = median_filter_2d(&noisy_image, 3, &center_weighted_config)....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 100: `let image_hybrid = hybrid_median_filter_2d(&noisy_image, 5, &standard_config).un...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 113: `100.0 * (image_mse_noisy - image_mse_standard) / image_mse_noisy`
  - **Fix**: Division without zero check - use safe_divide()
- Line 118: `100.0 * (image_mse_noisy - image_mse_adaptive) / image_mse_noisy`
  - **Fix**: Division without zero check - use safe_divide()
- Line 123: `100.0 * (image_mse_noisy - image_mse_weighted) / image_mse_noisy`
  - **Fix**: Division without zero check - use safe_divide()
- Line 128: `100.0 * (image_mse_noisy - image_mse_hybrid) / image_mse_noisy`
  - **Fix**: Division without zero check - use safe_divide()
- Line 146: `median_filter_color(&noisy_color_image, 3, &standard_config, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 147: `let color_vector = median_filter_color(&noisy_color_image, 3, &standard_config, ...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 158: `100.0 * (color_mse_noisy - color_mse_channel) / color_mse_noisy`
  - **Fix**: Division without zero check - use safe_divide()
- Line 163: `100.0 * (color_mse_noisy - color_mse_vector) / color_mse_noisy`
  - **Fix**: Division without zero check - use safe_divide()
- Line 215: `let edge_reflect = median_filter_2d(&noisy_image, 3, &reflect_config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 216: `let edge_nearest = median_filter_2d(&noisy_image, 3, &nearest_config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 217: `let edge_constant = median_filter_2d(&noisy_image, 3, &constant_config).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 218: `let edge_wrap = median_filter_2d(&noisy_image, 3, &wrap_config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 235: `let filtered = median_filter_2d(&noisy_image, size, &standard_config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 261: `clean_signal[i] = 2.0 + (i - 300) as f64 / 100.0 * 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 264: `clean_signal[i] = 3.0 + (i - 350) as f64 / 50.0 * std::f64::consts::PI;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 265: `clean_signal[i] = 3.0 + (clean_signal[i].sin() + 1.0) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 304: `let x = j as f64 / size as f64 * 2.0 - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 305: `let y = i as f64 / size as f64 * 2.0 - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 324: `clean_image[[i, j]] = 0.3 * (x + y + 2.0) / 4.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 364: `let x = j as f64 / size as f64 * 2.0 - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 365: `let y = i as f64 / size as f64 * 2.0 - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 393: `clean_image[[i, j, 0]] = 0.3 * (x + 1.0) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 394: `clean_image[[i, j, 1]] = 0.3 * (y + 1.0) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 395: `clean_image[[i, j, 2]] = 0.3 * ((1.0 - x) + 1.0) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 454: `sum_squared_diff / signal1.len() as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 473: `sum_squared_diff / (height * width) as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 494: `sum_squared_diff / (height * width * channels) as f64`
  - **Fix**: Division without zero check - use safe_divide()

### examples/memory_efficient_stft_example.rs

7 issues found:

- Line 55: `let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 163: `n as f64 * 8.0 / 1_000_000.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 169: `let t = i as f64 / fs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 232: `duration / processing_time.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 279: `let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 346: `let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 396: `sequential_time.as_secs_f64() / parallel_time.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()

### examples/meyer_wavelet_example.rs

6 issues found:

- Line 11: `let signal = chirp(&t, 0.0, 1.0, 100.0, "linear", 0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 21: `let coeffs = wavedec(&noisy_signal, Wavelet::Meyer, Some(3), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 46: `let denoised_signal = waverec(&denoised_coeffs, Wavelet::Meyer).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 164: `let coeffs = wavedec(noisy_signal, wavelet, Some(3), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 173: `let denoised = waverec(&denoised_coeffs, wavelet).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 181: `/ signal.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/minimum_phase_example.rs

7 issues found:

- Line 25: `let min_phase_b = minimum_phase(&original_b, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 29: `let frequencies = vec![0.0, PI / 4.0, PI / 2.0, 3.0 * PI / 4.0, PI];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 56: `let still_min_b = minimum_phase(&already_min_b, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 77: `let ct_min_b = minimum_phase(&ct_b, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 85: `let b_min = minimum_phase(&b_non_min, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 92: `let gd_orig = group_delay(&b_non_min, &a, &freqs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 93: `let gd_min = group_delay(&b_min, &a, &freqs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/nonlocal_means_example.rs

19 issues found:

- Line 26: `let denoised_signal = nlm_denoise_1d(&noisy_signal, &basic_config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 59: `let denoised_standard = nlm_denoise_2d(&noisy_image, &standard_config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 60: `let denoised_fast = nlm_denoise_2d(&noisy_image, &fast_config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 61: `let denoised_block = nlm_block_matching_2d(&noisy_image, &standard_config, 16).u...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 98: `let denoised_multiscale = nlm_multiscale_2d(&noisy_image, &standard_config, 2).u...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 112: `let denoised_color = nlm_color_image(&noisy_color_image, &standard_config).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 159: `let denoised = nlm_denoise_2d(&noisy_image, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 171: `let denoised = nlm_denoise_2d(&noisy_image, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 206: `let x = (i as f64 - edge as f64) / smooth_width as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 240: `let x = j as f64 / size as f64 * 2.0 - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 241: `let y = i as f64 / size as f64 * 2.0 - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 268: `clean_image.mapv_inplace(|x| (x - min_val) / (max_val - min_val));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 298: `let x = j as f64 / size as f64 * 2.0 - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 299: `let y = i as f64 / size as f64 * 2.0 - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 302: `clean_image[[i, j, 0]] = 0.5 * (1.0 - (x * x + y * y).sqrt().min(1.0));`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 305: `clean_image[[i, j, 1]] = if ((j as f64 / 8.0).floor() as i32) % 2 == 0 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 375: `10.0 * (signal_power / noise_power).log10()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 400: `10.0 * (signal_power / noise_power).log10()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 427: `10.0 * (signal_power / noise_power).log10()`
  - **Fix**: Division without zero check - use safe_divide()

### examples/parametric_spectral_example.rs

16 issues found:

- Line 49: `let duration = n_samples as f64 / fs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 127: `let freqs = Array1::linspace(0.0, fs / 2.0, nfft / 2 + 1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 182: `let freqs = Array1::linspace(0.0, fs / 2.0, nfft / 2 + 1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 235: `periodogram(signal.as_slice().unwrap(), Some(fs), None, None, None, None).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 354: `let freqs = Array1::linspace(0.0, fs / 2.0, nfft / 2 + 1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 358: `estimate_ar(&signal, ar_order, ARMethod::Burg).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 367: `ar_spectrum(&ar_only_coeffs, ar_only_variance, &freqs, fs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 372: `periodogram(signal.as_slice().unwrap(), Some(fs), None, None, None, None)`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 373: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 374: `let pxx_db: Vec<f64> = pxx_periodogram[..(nfft / 2 + 1)]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 436: `let (ar_coeffs, _, variance) = estimate_ar(signal, opt_order, method).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 440: `let freqs = Array1::linspace(0.0, fs / 2.0, nfft / 2 + 1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 441: `let psd = ar_spectrum(&ar_coeffs, variance, &freqs, fs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 446: `periodogram(signal.as_slice().unwrap(), Some(fs), None, None, None, None)`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 447: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 448: `let pxx_db: Vec<f64> = pxx_periodogram[..(nfft / 2 + 1)]`
  - **Fix**: Division without zero check - use safe_divide()

### examples/phase_vocoder_example.rs

6 issues found:

- Line 150: `let t = i as f64 / sample_rate;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 151: `let duration = samples as f64 / sample_rate;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 154: `let _freq = start_freq + (end_freq - start_freq) * t / duration;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 158: `2.0 * PI * (start_freq * t + (end_freq - start_freq) * t * t / (2.0 * duration))...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 169: `(sum_squared / signal.len() as f64).sqrt()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 169: `(sum_squared / signal.len() as f64).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation

### examples/reassigned_spectrogram_example.rs

21 issues found:

- Line 57: `let duration = n_samples as f64 / fs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 68: `let rate = (f1 - f0) / duration;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 72: `let center = duration / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 75: `let gaussian = (-((ti - center) / width).powi(2)).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 96: `let _win = window::hann(window_size, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 100: `signal.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 110: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 116: `let mut spectrogram = Array2::zeros((n_fft / 2 + 1, stft_complex.len()));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 138: `config.window = Array1::from(window::hann(256, true).unwrap());`
  - **Fix**: Handle array creation errors properly
- Line 145: `let result = reassigned_spectrogram(signal, config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 162: `config.window = Array1::from(window::hann(256, true).unwrap());`
  - **Fix**: Handle array creation errors properly
- Line 169: `let result = smoothed_reassigned_spectrogram(signal, config, 3).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 184: `let duration = n_samples as f64 / fs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 206: `let _win = window::hann(window_size, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 209: `signal.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 219: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 225: `let mut spectrogram = Array2::zeros((n_fft / 2 + 1, stft_complex.len()));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 234: `config.window = Array1::from(window::hann(window_size, true).unwrap());`
  - **Fix**: Handle array creation errors properly
- Line 240: `let reassigned_result = reassigned_spectrogram(&signal, config.clone()).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 241: `let smoothed_result = smoothed_reassigned_spectrogram(&signal, config, 3).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 306: `let time = i as f64 / 2048.0;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/remez_filter_example.rs

8 issues found:

- Line 20: `let h_lp = remez(numtaps, &bands, &desired, Some(&weights), None, None).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 32: `let h_fft = fft(&h_padded, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 33: `let _freqs: Vec<f64> = (0..nfft / 2).map(|i| i as f64 / nfft as f64).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 66: `let h_bp = remez(numtaps, &bands, &desired, Some(&weights), None, None).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 69: `let center = numtaps / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 83: `let h_mb = remez(numtaps, &bands, &desired, None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 89: `for i in 0..(numtaps / 2) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 103: `let h_diff = remez(numtaps, &bands, &desired, None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/signal_bspline_filtering.rs

2 issues found:

- Line 78: `/ signal.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 107: `/ signal.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/signal_interpolation.rs

28 issues found:

- Line 39: `signal[i] = (x[i] * PI / 2.0).sin() + 0.5 * (x[i] * PI).cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 66: `let min_len = signals.iter().map(|(_, data)| data.len()).min().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 124: `reference[i] = (x[i] * PI / 2.0).sin() + 0.5 * (x[i] * PI).cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 169: `let linear_mse = linear_sse / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 170: `let spline_mse = spline_sse / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 171: `let spectral_mse = spectral_sse / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 286: `let mse = sse / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 321: `let nyquist = n_samples / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 329: `let normal = Normal::<f64>::new(0.0, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 345: `spectrum[n_samples / 2] = Complex64::new(normal.sample(&mut rng).abs() as f64, 0...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 356: `let scale = 1.0 / n_samples as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 424: `let linear_mse = linear_sse / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 425: `let spline_mse = spline_sse / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 426: `let sinc_mse = sinc_sse / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 427: `let spectral_mse = spectral_sse / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 463: `let x = (j as f64) / (n_cols as f64) * 3.0 - 1.5;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 464: `let y = (i as f64) / (n_rows as f64) * 3.0 - 1.5;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 466: `let r1 = (x * x + y * y).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 467: `let r2 = ((x + 1.0) * (x + 1.0) + y * y).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 468: `let r3 = ((x - 1.0) * (x - 1.0) + y * y).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 470: `image[[i, j]] = 3.0 * (1.0 - r1).powi(2) * (-r1 / 3.0).exp()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 471: `- 10.0 * (r1 / 5.0 - r1.powi(3)) * (-r1 * r1).exp()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 472: `- 1.0 / 3.0 * (-r2 * r2).exp()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 473: `+ 1.0 / 5.0 * (-r3 * r3).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 549: `let linear_mse = linear_sse / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 550: `let spline_mse = spline_sse / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 551: `let nearest_mse = nearest_sse / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 653: `let mse = sse / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/signal_separation_example.rs

4 issues found:

- Line 20: `.map(|i| i as f64 / sample_rate)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 173: `reconstructed_energy / original_energy`
  - **Fix**: Division without zero check - use safe_divide()
- Line 184: `let mean_square: f64 = signal.iter().map(|&x| x * x).sum::<f64>() / signal.len()...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 185: `mean_square.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation

### examples/sparse_recovery_example.rs

28 issues found:

- Line 79: `compressed_sensing_recover(&noisy_y, &phi, SparseRecoveryMethod::OMP, &config).u...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 84: `vector_norm(&diff.view(), 2).unwrap() / vector_norm(&original_signal.view(), 2)....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 88: `measure_sparsity(&original_signal, 1e-6).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 92: `measure_sparsity(&recovered_signal, 1e-6).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 115: `let t = i as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 137: `100.0 * missing_count as f64 / n as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 150: `recover_missing_samples(&observed_signal, SparseRecoveryMethod::OMP, &config).un...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 163: `let rmse = (error_sum / count as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 163: `let rmse = (error_sum / count as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 189: `let t = i as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 201: `let signal_power = clean_signal.mapv(|v| v * v).sum() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 202: `let noise_power = (&clean_signal - &noisy_signal).mapv(|v| v * v).sum() / n as f...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 203: `let snr_before = 10.0 * (signal_power / noise_power).log10();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 223: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 226: `let noise_power_after = (&clean_signal - &denoised_signal).mapv(|v| v * v).sum()...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 227: `let snr_after = 10.0 * (signal_power / noise_power_after).log10();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 235: `let rmse_before = vector_norm(&diff_before.view(), 2).unwrap() / (n as f64).sqrt...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 236: `let rmse_after = vector_norm(&diff_after.view(), 2).unwrap() / (n as f64).sqrt()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 303: `let recovered_signal = compressed_sensing_recover(&noisy_y, &phi, method, &confi...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 309: `let recovery_error = vector_norm(&diff.view(), 2).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 310: `/ vector_norm(&original_signal.view(), 2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 313: `let sparsity = measure_sparsity(&recovered_signal, 1e-6).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 334: `let x = i as f64 / n_rows as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 335: `let y = j as f64 / n_cols as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 365: `100.0 * missing_count as f64 / (n_rows * n_cols) as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 383: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 398: `let rmse = (error_sum / count as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 398: `let rmse = (error_sum / count as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation

### examples/spectral_descriptors.rs

3 issues found:

- Line 26: `let t = i as f64 / fs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 94: `let t = i as f64 / fs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 124: `let t = i as f64 / fs;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/stft_example.rs

4 issues found:

- Line 24: `let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 35: `window_length as f64 * 1000.0 / fs`
  - **Fix**: Division without zero check - use safe_divide()
- Line 40: `hop_size as f64 * 1000.0 / fs`
  - **Fix**: Division without zero check - use safe_divide()
- Line 97: `let spec_mean = spectrogram.sum() / (spectrogram.shape()[0] * spectrogram.shape(...`
  - **Fix**: Division without zero check - use safe_divide()

### examples/streaming_stft_example.rs

15 issues found:

- Line 55: `let t = i as f64 / sample_rate;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 73: `sample_rate / config.frame_length as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 88: `let peak_frequency = peak_bin as f64 * sample_rate / config.frame_length as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 111: `stats.latency_samples as f64 / sample_rate,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 148: `let block_time = block_num as f64 * block_size as f64 / sample_rate;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 153: `let t = block_time + i as f64 / sample_rate;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 154: `let freq = start_freq + (end_freq - start_freq) * t / sweep_duration;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 186: `let peak_frequency = peak_bin as f64 * sample_rate / rt_config.frame_length as f...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 246: `.map(|j| (2.0 * PI * test_tone * j as f64 / sample_rate).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 262: `peak_bin as f64 * sample_rate / low_latency_config.frame_length as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 292: `let t = i as f64 / sample_rate;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 315: `let frequency_resolution = sample_rate / batch_config.frame_length as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 337: `peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // Sort by magnitude`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 398: `let freq_resolution = sample_rate / config.frame_length as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 399: `let time_resolution = config.hop_length as f64 / sample_rate * 1000.0;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/swt2d_edge_detection.rs

4 issues found:

- Line 12: `let circle_center = (size as f64 / 2.0, size as f64 / 2.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 27: `let distance = (x * x + y * y).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 70: `let ratio = count_nonzero(&edges_clean, 0.1 * max_edge_value) as f64 / (size * s...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 101: `(decomp.detail_h[[i, j]].powi(2) + decomp.detail_v[[i, j]].powi(2)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation

### examples/swt2d_example.rs

1 issues found:

- Line 152: `100.0 * (mse_noisy - mse_denoised) / mse_noisy`
  - **Fix**: Division without zero check - use safe_divide()

### examples/swt_example.rs

5 issues found:

- Line 11: `let t = (0..1000).map(|i| i as f64 / fs).collect::<Vec<f64>>();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 12: `let signal = chirp(&t, 0.0, 1.0, 100.0, "linear", 0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 22: `let (details, approx) = swt(&noisy_signal, Wavelet::DB(4), 3, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 36: `let denoised_signal = iswt(&modified_details, &approx, Wavelet::DB(4)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 39: `let reconstructed_signal = iswt(&details, &approx, Wavelet::DB(4)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/synchrosqueezed_example.rs

12 issues found:

- Line 47: `let rate_1 = (f1_1 - f0_1) / duration;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 57: `let k = ((f1_3 / f0_3) as f64).powf(1.0 / duration);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 57: `let k = ((f1_3 / f0_3) as f64).powf(1.0 / duration);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 63: `let _freq = f0_3 * k.powf(ti_adj);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 64: `let phase = 2.0 * PI * f0_3 * (k.powf(ti_adj) - 1.0) / k.ln();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 64: `let phase = 2.0 * PI * f0_3 * (k.powf(ti_adj) - 1.0) / k.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 64: `let phase = 2.0 * PI * f0_3 * (k.powf(ti_adj) - 1.0) / k.ln();`
  - **Fix**: Mathematical operation .powf( without validation
- Line 97: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 126: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 159: `let mid_idx = n_points / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 164: `first_section.iter().map(|&(_, f)| f).sum::<f64>() / first_section.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 166: `second_section.iter().map(|&(_, f)| f).sum::<f64>() / second_section.len() as f6...`
  - **Fix**: Division without zero check - use safe_divide()

### examples/sysid_example.rs

5 issues found:

- Line 22: `let t = Array1::linspace(0.0, (n - 1) as f64 / fs, n);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 29: `let t_slice = t.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 37: `let _dt = 1.0 / fs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 96: `let mid_idx = freq_result.coherence.len() / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 218: `let avg_error = estimation_errors.iter().sum::<f64>() / estimation_errors.len() ...`
  - **Fix**: Division without zero check - use safe_divide()

### examples/test_windows.rs

17 issues found:

- Line 10: `let hamming_win = hamming(n, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 15: `let hann_win = hann(n, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 20: `let bartlett_win = bartlett(n, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 35: `.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 36: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 46: `let _center_idx = (n - 1) as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 55: `let idx1 = n / 2 - 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 56: `let idx2 = n / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 58: `0.54 - 0.46 * (2.0 * std::f64::consts::PI * idx1 as f64 / (n - 1) as f64).cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 60: `0.54 - 0.46 * (2.0 * std::f64::consts::PI * idx2 as f64 / (n - 1) as f64).cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 72: `let idx1 = n / 2 - 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 73: `let idx2 = n / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 75: `0.5 * (1.0 - (2.0 * std::f64::consts::PI * idx1 as f64 / (n - 1) as f64).cos());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 77: `0.5 * (1.0 - (2.0 * std::f64::consts::PI * idx2 as f64 / (n - 1) as f64).cos());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 87: `let idx = n / 2 - 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 88: `let expected = 2.0 * idx as f64 / (n - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 96: `let idx = n / 2;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/test_wvd.rs

1 issues found:

- Line 17: `let wvd = wigner_ville(&signal, config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/total_variation_example.rs

24 issues found:

- Line 36: `let denoised_standard = tv_denoise_1d(&noisy_signal, weight, &standard_config).u...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 37: `let denoised_anisotropic = tv_denoise_1d(&noisy_signal, weight, &anisotropic_con...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 72: `let denoised_image = tv_denoise_2d(&noisy_image, image_weight, &standard_config)...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 74: `tv_denoise_2d(&noisy_image, image_weight, &anisotropic_config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 105: `let bregman_1d = tv_bregman_1d(&noisy_signal, weight, 3, &standard_config).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 106: `let bregman_2d = tv_bregman_2d(&noisy_image, image_weight, 3, &standard_config)....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 134: `tv_denoise_color(&noisy_color_image, 0.2, &standard_config, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 136: `tv_denoise_color(&noisy_color_image, 0.2, &standard_config, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 194: `let inpainted_image = tv_inpaint(&corrupted_image, 0.1, &standard_config).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 225: `let denoised = tv_denoise_2d(&noisy_image, w, &standard_config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 252: `clean_signal[i] = 0.5 + (i - 300) as f64 / 100.0 * 0.5;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 280: `let x = j as f64 / size as f64 * 2.0 - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 281: `let y = i as f64 / size as f64 * 2.0 - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 310: `clean_image.mapv_inplace(|x| (x - min_val) / (max_val - min_val));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 340: `let x = j as f64 / size as f64 * 2.0 - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 341: `let y = i as f64 / size as f64 * 2.0 - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 420: `let line_y = height / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 428: `let line_x = width / 3;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 436: `let rect_x = 3 * width / 4;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 437: `let rect_y = height / 4;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 495: `10.0 * (signal_power / noise_power).log10()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 515: `10.0 * (signal_power / noise_power).log10()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 537: `10.0 * (signal_power / noise_power).log10()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 562: `10.0 * (1.0 / mse).log10()`
  - **Fix**: Division without zero check - use safe_divide()

### examples/wavelet2d_example.rs

1 issues found:

- Line 112: `original_nonzero as f64 / thresholded_nonzero as f64`
  - **Fix**: Division without zero check - use safe_divide()

### examples/wavelet_image_processing.rs

17 issues found:

- Line 141: `let center_x = (width / 2) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 142: `let center_y = (height / 2) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 143: `let max_radius = (width.min(height) / 2) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 150: `let radius = (dx * dx + dy * dy).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 151: `let normalized_radius = radius / max_radius;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 169: `let h_edge = height / 3;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 172: `let v_edge = width * 2 / 3;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 202: `let val3 = ((x as f64 - width as f64 / 2.0).powi(2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 203: `+ (y as f64 - height as f64 / 2.0).powi(2))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 204: `.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 231: `let mean = sum / (image.len() as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 232: `let variance = sum_sq / (image.len() as f64) - mean * mean;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 233: `let std_dev = variance.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 242: `100.0 * nonzero_count as f64 / image.len() as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 260: `sum_squared_error / n`
  - **Fix**: Division without zero check - use safe_divide()
- Line 276: `20.0 * (data_range / mse.sqrt()).log10()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 276: `20.0 * (data_range / mse.sqrt()).log10()`
  - **Fix**: Mathematical operation .sqrt() without validation

### examples/wavelet_packet_2d.rs

6 issues found:

- Line 14: `let center_x = size as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 15: `let center_y = size as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 16: `let radius = size as f64 / 4.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 26: `let distance = (x * x + y * y).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 52: `let packet = full_decomp.get_packet(1, row, col).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 111: `100.0 * (1.0 - (total_count as f64 / (1 + 4 + 16 + full_nodes_at_level3) as f64)...`
  - **Fix**: Division without zero check - use safe_divide()

### examples/wavelet_visualization.rs

12 issues found:

- Line 33: `let base = if (i / 8 + j / 8) % 2 == 0 { 1.0 } else { 0.3 };`
  - **Fix**: Division without zero check - use safe_divide()
- Line 37: `let ci = i as f64 - size as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 38: `let cj = j as f64 - size as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 39: `let distance = (ci * ci + cj * cj).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 42: `let gradient = 0.5 * (i as f64 / size as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 45: `image[[i, j]] = 0.8 + 0.2 * distance / circle_radius;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 79: `100.0 * energy.horizontal.unwrap() / energy.total,`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 80: `energy.horizontal.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 84: `100.0 * energy.vertical.unwrap() / energy.total,`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 85: `energy.vertical.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 89: `100.0 * energy.diagonal.unwrap() / energy.total,`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 90: `energy.diagonal.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/wiener_filter_example.rs

20 issues found:

- Line 29: `let denoised_basic = wiener_filter(&noisy_signal, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 56: `let denoised_freq = wiener_filter_freq(&noisy_signal, &freq_config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 57: `let denoised_time = wiener_filter_time(&noisy_signal, &time_config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 84: `let denoised_iter = iterative_wiener_filter(&noisy_signal, &iter_config).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 102: `let denoised_spec1 = spectral_subtraction(&noisy_signal, None, Some(1.0), Some(0...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 103: `let denoised_spec2 = spectral_subtraction(&noisy_signal, None, Some(2.0), Some(0...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 123: `let denoised_psd = psd_wiener_filter(&noisy_signal, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 144: `let nonstat_wiener = wiener_filter(&noisy_signal2, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 146: `kalman_wiener_filter(&noisy_signal2, process_var, measurement_var).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 175: `let denoised_image = wiener_filter_2d(&noisy_image, None, Some([5, 5])).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 200: `let t = Array1::linspace(0.0, (n_samples as f64 - 1.0) / sampling_rate, n_sample...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 204: `waveforms::chirp(t.as_slice().unwrap(), 10.0, 1.0, 100.0, "linear", 0.0).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 224: `let t = Array1::linspace(0.0, (n_samples as f64 - 1.0) / sampling_rate, n_sample...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 238: `let noise_level = 0.2 + 0.8 * (1.0 - (i as f64 / n_samples as f64 * 2.0 - 1.0).p...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 259: `let x = j as f64 / width as f64 * 2.0 - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 260: `let y = i as f64 / height as f64 * 2.0 - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 263: `let distance = (x.powi(2) + y.powi(2)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 277: `clean_image.mapv_inplace(|x| (x - min_val) / (max_val - min_val));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 315: `10.0 * (signal_power / noise_power).log10()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 338: `10.0 * (signal_power / noise_power).log10()`
  - **Fix**: Division without zero check - use safe_divide()

### examples/wigner_ville_example.rs

11 issues found:

- Line 59: `let duration = n_samples as f64 / fs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 70: `let rate = (f1 - f0) / duration;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 74: `let center = duration / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 77: `let gaussian = (-((ti - center) / width).powi(2)).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 105: `let wvd = wigner_ville(signal, config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 149: `let spwvd = smoothed_pseudo_wigner_ville(signal, &time_win, &freq_win, config).u...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 168: `let duration = n_samples as f64 / fs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 188: `let wvd = wigner_ville(&signal, config.clone()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 197: `let spwvd = smoothed_pseudo_wigner_ville(&signal, &time_win, &freq_win, config)....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 210: `let duration = n_samples as f64 / fs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 228: `let xwvd_complex = cross_wigner_ville(&signal_sin, &signal_cos, config).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/window_comparison.rs

1 issues found:

- Line 49: `let mean = sum / window.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/wpt_example.rs

5 issues found:

- Line 12: `let t = (0..1024).map(|i| i as f64 / fs).collect::<Vec<f64>>();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 13: `let signal = chirp(&t, 0.0, 1.0, 100.0, "linear", 0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 24: `let wpt = wp_decompose(&noisy_signal, Wavelet::DB(4), level, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 60: `let denoised_signal = reconstruct_from_nodes(&wpt, &nodes).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 108: `let best_basis_signal = reconstruct_from_nodes(&wpt, &selected_nodes).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/z_domain_design_example.rs

3 issues found:

- Line 112: `h_num += coeff * z.powf(-(i as f64));`
  - **Fix**: Mathematical operation .powf( without validation
- Line 115: `h_den += coeff * z.powf(-(i as f64));`
  - **Fix**: Mathematical operation .powf( without validation
- Line 118: `let response = h_num / h_den;`
  - **Fix**: Division without zero check - use safe_divide()

### src/adaptive.rs

68 issues found:

- Line 313: `*p_elem = (*p_elem - kxp) / self.lambda;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 471: `let normalized_step = self.step_size / (input_power + self.epsilon);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 913: `.map(|c| c.re / self.block_size as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 962: `.map(|c| c.re / self.block_size as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1145: `let normalized_step = self.step_size / normalization;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1168: `self.update_count as f64 / self.sample_count as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1248: `let factor = aug_matrix[k][i] / aug_matrix[i][i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1275: `let lms = LmsFilter::new(4, 0.01, 0.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1286: `let mut lms = LmsFilter::new(2, 0.1, 0.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1289: `let (output, error, _mse) = lms.adapt(1.0, 0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1301: `let mut lms = LmsFilter::new(2, 0.05, 0.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1306: `let (outputs, errors, _mse) = lms.adapt_batch(&inputs, &desired).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1319: `let mut lms = LmsFilter::new(3, 0.01, 0.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1343: `let (_outputs, _errors, _mse) = lms.adapt_batch(&inputs, &desired).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1361: `let rls = RlsFilter::new(3, 0.99, 100.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1373: `let mut rls = RlsFilter::new(2, 0.99, 100.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1375: `let (output, error, _mse) = rls.adapt(1.0, 0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1384: `let nlms = NlmsFilter::new(4, 0.5, 1e-6).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1395: `let mut nlms = NlmsFilter::new(2, 0.5, 1e-6).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1397: `let (output, error, _mse) = nlms.adapt(1.0, 0.3).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1430: `let mut lms = LmsFilter::new(2, 0.05, 0.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1431: `let mut rls = RlsFilter::new(2, 0.99, 100.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1444: `let (_out_lms, err_lms, _) = lms.adapt(input, desired).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1445: `let (_out_rls, err_rls, _) = rls.adapt(input, desired).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1452: `let lms_final_error = lms_errors.iter().rev().take(10).sum::<f64>() / 10.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1453: `let rls_final_error = rls_errors.iter().rev().take(10).sum::<f64>() / 10.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1471: `let vs_lms = VsLmsFilter::new(4, 0.01, 0.05).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1484: `let mut vs_lms = VsLmsFilter::new(2, 0.1, 0.01).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1486: `let (output, error, _mse) = vs_lms.adapt(1.0, 0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1496: `vs_lms.adapt(1.0, 0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1505: `let apa = ApaFilter::new(4, 3, 0.1, 0.01).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1516: `let mut apa = ApaFilter::new(2, 2, 0.1, 0.01).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1519: `let (output, error, _mse) = apa.adapt(&input, 0.3).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1532: `let fdlms = FdlmsFilter::new(8, 0.01, 0.999).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1542: `let mut fdlms = FdlmsFilter::new(4, 0.01, 0.999).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1546: `let (outputs, errors) = fdlms.adapt_block(&inputs, &desired).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1560: `let lmf = LmfFilter::new(4, 0.01).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1570: `let mut lmf = LmfFilter::new(2, 0.01).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1572: `let (output, error, _mse) = lmf.adapt(1.0, 0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1580: `lmf.adapt(1.0, 0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1584: `let (final_output, _, _) = lmf.adapt(1.0, 0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1590: `let sm_lms = SmLmsFilter::new(4, 0.1, 0.1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1601: `let mut sm_lms = SmLmsFilter::new(2, 0.1, 0.05).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1604: `let (output, error, _mse) = sm_lms.adapt(1.0, 0.01).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1613: `sm_lms.adapt(1.0, 0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1622: `let mut sm_lms = SmLmsFilter::new(3, 0.1, 0.1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1630: `sm_lms.adapt(input, target_error).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1645: `let mut lms = LmsFilter::new(3, 0.01, 0.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1646: `let mut vs_lms = VsLmsFilter::new(3, 0.01, 0.05).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1647: `let mut nlms = NlmsFilter::new(3, 0.5, 1e-6).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1648: `let mut lmf = LmfFilter::new(3, 0.001).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1667: `let (_, err_lms, _) = lms.adapt(input, desired).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1668: `let (_, err_vs_lms, _) = vs_lms.adapt(input, desired).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1669: `let (_, err_nlms, _) = nlms.adapt(input, desired).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1670: `let (_, err_lmf, _) = lmf.adapt(input, desired).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1681: `lms_errors.iter().rev().take(final_window).sum::<f64>() / final_window as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1683: `vs_lms_errors.iter().rev().take(final_window).sum::<f64>() / final_window as f64...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1685: `nlms_errors.iter().rev().take(final_window).sum::<f64>() / final_window as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1687: `lmf_errors.iter().rev().take(final_window).sum::<f64>() / final_window as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1724: `let solution = solve_linear_system_small(&matrix, &rhs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1742: `let mut vs_lms = VsLmsFilter::new(3, 0.01, 0.05).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1743: `let mut apa = ApaFilter::new(3, 2, 0.1, 0.01).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1744: `let mut lmf = LmfFilter::new(3, 0.01).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1745: `let mut sm_lms = SmLmsFilter::new(3, 0.1, 0.1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1750: `vs_lms.adapt(input, 0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1751: `apa.adapt(&[input, input * 0.5, input * 0.2], 0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1752: `lmf.adapt(input, 0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1753: `sm_lms.adapt(input, 0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/advanced_filter.rs

29 issues found:

- Line 313: `let weight = response.weights[i].sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 324: `.map(|(&mag, &weight)| mag * weight.sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 353: `.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 393: `let weight = response.weights[i].sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 442: `error = error.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 487: `let delta_p = (10.0_f64.powf(spec.passband_ripple / 20.0) - 1.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 487: `let delta_p = (10.0_f64.powf(spec.passband_ripple / 20.0) - 1.0)`
  - **Fix**: Mathematical operation .powf( without validation
- Line 488: `/ (10.0_f64.powf(spec.passband_ripple / 20.0) + 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 488: `/ (10.0_f64.powf(spec.passband_ripple / 20.0) + 1.0);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 489: `let delta_s = 10.0_f64.powf(-spec.stopband_attenuation / 20.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 489: `let delta_s = 10.0_f64.powf(-spec.stopband_attenuation / 20.0);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 504: `let normalized_width = transition_width / spec.sample_rate;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 509: `((a - 13.0) / (14.6 * normalized_width)).ceil() as usize`
  - **Fix**: Division without zero check - use safe_divide()
- Line 511: `((a - 7.95) / (14.36 * normalized_width)).ceil() as usize`
  - **Fix**: Division without zero check - use safe_divide()
- Line 513: `(0.9222 / normalized_width).ceil() as usize`
  - **Fix**: Division without zero check - use safe_divide()
- Line 524: `let nyquist = spec.sample_rate / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 527: `let passband_norm: Vec<f64> = spec.passband_freqs.iter().map(|&f| f / nyquist).c...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 528: `let stopband_norm: Vec<f64> = spec.stopband_freqs.iter().map(|&f| f / nyquist).c...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 618: `let idx = (i * (grid_len - 1)) / (num_extremal - 1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 647: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 649: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 736: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 778: `/ (freq_points[upper_idx] - freq_points[lower_idx]);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 829: `let estimated_order = estimate_filter_order(&spec).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 847: `let (frequencies, desired, weights) = create_design_grid(&spec, &config).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 873: `let result = arbitrary_magnitude_design(&response, 16, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 893: `let result = least_squares_design(&response, 8).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 919: `let (frequencies, response) = compute_frequency_response(&coefficients, 64).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 939: `interpolate_response(&freq_points, &response_values, &new_freq_points).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/bss/fastica.rs

9 issues found:

- Line 44: `let normal = Normal::new(0.0, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 61: `Box::new(|x: f64| x * (-x * x / 2.0).exp()),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 62: `Box::new(|x: f64| (-x * x / 2.0).exp() * (1.0 - x * x)),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 124: `new_wp[i] = sum_gx_x / (n_samples as f64) - g_prime_sum * wp[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 137: `let norm = (new_wp.dot(&new_wp)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 180: `let gradient = gx.dot(&signals.t()) / (n_samples as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 181: `- Array2::<f64>::eye(n_components) * w.mapv(|x: f64| g_prime(x)).mean().unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 200: `d_inv_sqrt[[i, i]] = 1.0 / eigvals[i].sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 200: `d_inv_sqrt[[i, i]] = 1.0 / eigvals[i].sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation

### src/bss/ica.rs

1 issues found:

- Line 47: `let means = signals.mean_axis(Axis(1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/bss/infomax.rs

13 issues found:

- Line 41: `let normal = Normal::new(0.0, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 63: `let n_batches = n_samples / batch_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 81: `y_sigmoid[[i, j]] = 1.0 / (1.0 + (-y[[i, j]]).exp());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 88: `&block.dot(&y_sigmoid.dot(&x_batch.t())) * (learning_rate / batch_size as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 94: `let delta_w_avg = delta_w_sum / n_batches as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 101: `if delta_w_avg.mapv(|x: f64| x.abs()).mean().unwrap() < config.convergence_thres...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 142: `let normal = Normal::new(0.0, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 164: `let n_batches = n_samples / batch_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 188: `kurtosis = kurtosis / batch_size as f64 - 3.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 204: `1.0 - k.slice(s![i, ..]).mapv(|x: f64| x.powi(2)).mean().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 215: `let block = &eye - &k.dot(&y.t()) / batch_size as f64 + &k_prime;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 222: `let delta_w_avg = delta_w_sum / n_batches as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 229: `if delta_w_avg.mapv(|x: f64| x.abs()).mean().unwrap() < config.convergence_thres...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/bss/jade.rs

2 issues found:

- Line 111: `PI / 4.0 * (g21 >= 0.0) as i32 as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 113: `0.5 * (g21 / g22).atan()`
  - **Fix**: Division without zero check - use safe_divide()

### src/bss/joint.rs

9 issues found:

- Line 44: `let means = dataset.mean_axis(Axis(1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 54: `let cov = centered.dot(&centered.t()) / (n_samples as f64 - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 125: `s_inv[[i, i]] = 1.0 / s[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 157: `let means = signals.mean_axis(Axis(1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 170: `let cov = centered.dot(&centered.t()) / (n_samples as f64 - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 174: `let max_lag = 10.min(n_samples / 4);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 187: `cov_lagged[[i, j]] = sum / (n_samples as f64 - lag as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 221: `let theta = 0.5 * (num / denom).atan();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 273: `s_inv[[i, i]] = 1.0 / s[i];`
  - **Fix**: Division without zero check - use safe_divide()

### src/bss/kernel.rs

10 issues found:

- Line 35: `let means = signals.mean_axis(Axis(1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 57: `let normal = Normal::new(0.0, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 81: `gram[[i, j]] = (-diff * diff / (2.0 * kernel_width * kernel_width)).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 86: `let row_means = gram.mean_axis(Axis(0)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 87: `let col_means = gram.mean_axis(Axis(1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 88: `let total_mean = gram.mean().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 171: `gradient[[c, d]] = grad_cd / (n_samples * n_samples) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 192: `d_inv_sqrt[[i, i]] = 1.0 / eigvals[i].sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 192: `d_inv_sqrt[[i, i]] = 1.0 / eigvals[i].sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 226: `s_inv[[i, i]] = 1.0 / s[i];`
  - **Fix**: Division without zero check - use safe_divide()

### src/bss/memd.rs

5 issues found:

- Line 52: `let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 122: `let step = 1.0 / (idx2 - idx1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 150: `let diff = (&imf - &prev_imf).mapv(|x: f64| x.powi(2)).sum().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 151: `let norm = imf.mapv(|x: f64| x.powi(2)).sum().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 153: `if diff / norm < config.convergence_threshold {`
  - **Fix**: Division without zero check - use safe_divide()

### src/bss/mod.rs

21 issues found:

- Line 105: `let cov = signals.dot(&signals.t()) / (n_samples as f64 - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 121: `d_inv_sqrt[[i, i]] = 1.0 / eigvals[i].sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 121: `d_inv_sqrt[[i, i]] = 1.0 / eigvals[i].sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 155: `let mean = component.mean().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 156: `let var = component.mapv(|x: f64| (x - mean).powi(2)).sum() / (n_samples as f64 ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 161: `variances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 196: `let mean = signal.mean().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 197: `let std_dev = (signal.mapv(|x: f64| (x - mean).powi(2)).sum() / n_samples as f64...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 197: `let std_dev = (signal.mapv(|x: f64| (x - mean).powi(2)).sum() / n_samples as f64...`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 201: `normalized[[i, j]] = (signals[[i, j]] - mean) / std_dev;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 207: `let corr = normalized.dot(&normalized.t()) / (n_samples as f64 - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 245: `let x_bin_width = (x_max - x_min) / n_bins as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 246: `let y_bin_width = (y_max - y_min) / n_bins as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 255: `let x_bin = ((x[s] - x_min) / x_bin_width).floor() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 256: `let y_bin = ((y[s] - y_min) / y_bin_width).floor() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 279: `* (joint_hist[[xi, yi]] / (x_hist[xi] * y_hist[yi])).ln();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 279: `* (joint_hist[[xi, yi]] / (x_hist[xi] * y_hist[yi])).ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 305: `let means = signals.mean_axis(Axis(1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 315: `let cov = centered.dot(&centered.t()) / (n_samples as f64 - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 332: `ratios.push(eigvals[i] / eigvals[i + 1]);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 352: `Ok(n_signals / 2)`
  - **Fix**: Division without zero check - use safe_divide()

### src/bss/nmf.rs

4 issues found:

- Line 64: `let norm = w.slice(s![.., j]).mapv(|x: f64| x.powi(2)).sum().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 81: `h[[i, j]] *= w_t_v[[i, j]] / w_t_w_h[[i, j]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 93: `w[[i, j]] *= v_h_t[[i, j]] / w_h_h_t[[i, j]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 100: `let norm = w.slice(s![.., j]).mapv(|x: f64| x.powi(2)).sum().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation

### src/bss/pca.rs

4 issues found:

- Line 26: `let means = signals.mean_axis(Axis(1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 36: `let cov = centered.dot(&centered.t()) / (n_samples as f64 - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 50: `indices.sort_by(|&i, &j| eigvals[j].partial_cmp(&eigvals[i]).unwrap());`
  - **Fix**: Use .get() with proper bounds checking
- Line 71: `if cum_var / total_var >= config.variance_threshold {`
  - **Fix**: Division without zero check - use safe_divide()

### src/convolve.rs

7 issues found:

- Line 82: `let start_idx = (n_v - 1) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 235: `let pad_rows = n_rows_v / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 236: `let pad_cols = n_cols_v / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 292: `let result = convolve(&a, &v, "full").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 306: `let result = convolve(&a, &v, "same").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 319: `let result = convolve(&a, &v, "valid").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 332: `let result = correlate(&a, &v, "full").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/convolve_parallel.rs

4 issues found:

- Line 128: `let n_chunks = (na + step - 1) / step;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 193: `let start = (nv - 1) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 299: `"same" => (0, out_i as isize - (ker_rows / 2) as isize),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 308: `"same" => (0, out_j as isize - (ker_cols / 2) as isize),`
  - **Fix**: Division without zero check - use safe_divide()

### src/cqt.rs

37 issues found:

- Line 179: `1.0 / (2.0f64.powf(1.0 / config.bins_per_octave as f64) - 1.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 179: `1.0 / (2.0f64.powf(1.0 / config.bins_per_octave as f64) - 1.0)`
  - **Fix**: Mathematical operation .powf( without validation
- Line 184: `((config.f_max / config.f_min).log2() * config.bins_per_octave as f64).ceil() as...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 189: `frequencies[k] = config.f_min * 2.0f64.powf(k as f64 / config.bins_per_octave as...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 189: `frequencies[k] = config.f_min * 2.0f64.powf(k as f64 / config.bins_per_octave as...`
  - **Fix**: Mathematical operation .powf( without validation
- Line 240: `let n_bins = ((f_max / f_min).log2() * bins_per_octave as f64).ceil() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 245: `frequencies[k] = f_min * 2.0f64.powf(k as f64 / bins_per_octave as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 245: `frequencies[k] = f_min * 2.0f64.powf(k as f64 / bins_per_octave as f64);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 250: `let max_kernel_length = (window_scale * q * fs / f_min).ceil() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 267: `let kernel_length = (window_scale * q * fs / freq).ceil() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 279: `let center = (kernel_length - 1) as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 282: `let time = (n as f64 - center) / fs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 298: `.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 303: `padded_kernel[n] = kernel_values[n] / norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 381: `cqt[k] = sum / sparse_kernel.normalization;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 387: `let n_chunks = (n_signal as f64 / n_fft as f64).ceil() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 416: `cqt[k] += sum / sparse_kernel.normalization;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 440: `let n_frames = (n_signal as f64 / hop_size as f64).ceil() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 455: `times[frame] = (start + (end - start) / 2) as f64 / kernel.fs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 534: `magnitude[[i, j]] = 20.0 * (magnitude[[i, j]] / (reference + eps)).log10();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 629: `let times = cqt.times.as_ref().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 635: `n_fft / 2`
  - **Fix**: Division without zero check - use safe_divide()
- Line 772: `let hann = create_window("hann", length).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 805: `let q = 1.0 / (2.0f64.powf(1.0 / bins_per_octave as f64) - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 805: `let q = 1.0 / (2.0f64.powf(1.0 / bins_per_octave as f64) - 1.0);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 809: `compute_cqt_kernel(f_min, f_max, bins_per_octave, q, fs, "hann", None, true).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 812: `let expected_bins = ((f_max / f_min).log2() * bins_per_octave as f64).ceil() as ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 852: `let cqt_result = constant_q_transform(&signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 875: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 878: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 894: `let phase = 2.0 * PI * (110.0 * ti + (880.0 - 110.0) * ti * ti / (2.0 * duration...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 912: `let cqt_result = constant_q_transform(&signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 916: `((config.f_max / config.f_min).log2() * config.bins_per_octave as f64).ceil() as...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 917: `let n_frames = (n_samples as f64 / 512.0).ceil() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 924: `assert_eq!(cqt_result.times.unwrap().len(), n_frames);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 952: `let cqt_result = constant_q_transform(&signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 955: `let chroma = chromagram(&cqt_result, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/czt.rs

9 issues found:

- Line 46: `let arg = -2.0 * std::f64::consts::PI / m as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 127: `let arg = -2.0 * std::f64::consts::PI / m_val as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 292: `let inv_n = 1.0 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 309: `let points = czt_points(4, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 320: `let angle = -2.0 * std::f64::consts::PI * i as f64 / 4.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 336: `let czt_result = czt(&signal, None, None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 355: `let czt_result2 = czt(&signal2, None, None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 368: `let arg = -std::f64::consts::PI / 16.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 371: `let czt_result = czt(&signal, Some(8), Some(w), None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/deconvolution.rs

83 issues found:

- Line 93: `(padded_signal, padded_psf, (pad_len - n) / 2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 101: `let start = (n - psf.len()) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 147: `result_complex[i] = signal_complex[i] * h_conj / denom;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 157: `let scale = 1.0 / (pad_len as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 225: `(padded_signal, padded_psf, (pad_len - n) / 2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 233: `let start = (n - psf.len()) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 288: `result_complex[i] = signal_complex[i] * h_conj / denom;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 298: `let scale = 1.0 / (pad_len as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 367: `(padded_signal, padded_psf, (pad_len - n) / 2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 375: `let start = (n - psf.len()) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 392: `let signal_mean = padded_signal.sum() / (pad_len as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 412: `estimate.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 413: `normalized_psf.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 421: `correction[i] = padded_signal[i] / pred_val;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 426: `correction.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 427: `flipped_psf.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 437: `/ prev_estimate.mapv(|x| x.abs()).sum();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 510: `(padded_signal, padded_psf, (pad_len - n) / 2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 518: `let start = (n - psf.len()) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 550: `.max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 551: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 566: `(i as isize - peak_idx as isize + pad_len as isize / 2) as usize % pad_len;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 572: `let restoring_beam = create_gaussian_kernel((psf.len() / 2).max(3));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 574: `model.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 575: `restoring_beam.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 635: `(padded_signal, padded_psf, (pad_len - n) / 2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 643: `let start = (n - psf.len()) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 663: `model.fill(total_flux / (pad_len as f64));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 675: `model.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 676: `normalized_psf.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 684: `chi_squared += (diff * diff) / (noise_level * noise_level);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 702: `chi_grad += diff * normalized_psf[k] / (noise_level * noise_level);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 706: `let entropy_grad = -1.0 - model[i].ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 716: `model *= total_flux / model_sum;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 722: `(&model - &prev_model).mapv(|x| x.abs()).sum() / prev_model.mapv(|x| x.abs()).su...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 777: `(padded_signal, (pad_len - n) / 2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 823: `let start = (pad_len - psf_size) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 842: `/ prev_signal.mapv(|x| x.abs()).sum();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 845: `/ prev_psf.mapv(|x| x.abs()).sum();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 914: `(padded_image, padded_psf, pad_h / 2, pad_w / 2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 920: `let start_h = (height - psf_h) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 921: `let start_w = (width - psf_w) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 971: `result_complex[i] = image_complex[i] * h_conj / denom;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 981: `let scale = 1.0 / (pad_height * pad_width) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1050: `(padded_image, padded_psf, pad_h / 2, pad_w / 2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1056: `let start_h = (height - psf_h) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1057: `let start_w = (width - psf_w) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1076: `let image_mean = padded_image.sum() / (pad_height * pad_width) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1102: `correction[[i, j]] = padded_image[[i, j]] / pred_val;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1117: `/ prev_estimate.mapv(|x| x.abs()).sum();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1181: `(padded_image, padded_psf, pad_h / 2, pad_w / 2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1187: `let start_h = (height - psf_h) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1188: `let start_w = (width - psf_w) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1243: `let grad_mag = (dx * dx + dy * dy).sqrt() + eps;`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1247: `/ ((estimate[[i, j]] - estimate[[i, j - 1]]).powi(2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1249: `.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1250: `- dx / grad_mag;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1253: `/ ((estimate[[i, j]] - estimate[[i, j - 1]]).powi(2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1255: `.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1256: `- dy / grad_mag;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1279: `/ prev_estimate.mapv(|x| x.abs()).sum();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1339: `(padded_image, pad_h / 2, pad_w / 2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1389: `let start_h = (pad_height - psf_size_h) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1390: `let start_w = (pad_width - psf_size_w) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1420: `/ prev_image.mapv(|x| x.abs()).sum();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1423: `/ prev_psf.mapv(|x| x.abs()).sum();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1445: `let half_size = size as isize / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1453: `kernel[i] = (-x * x / two_sigma_sq).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1467: `let half_h = height as isize / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1468: `let half_w = width as isize / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1480: `kernel[[i, j]] = (-x * x / two_sigma_w_sq - y * y / two_sigma_h_sq).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1502: `signal.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1503: `kernel.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1625: `let log_min = min_param.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 1626: `let log_max = max_param.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 1627: `let step = (log_max - log_min) / (num_values - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1639: `let start = (n - psf.len()) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1677: `let filter_elem = h_abs_sq / (h_abs_sq + param);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1683: `result_complex[i] = signal_complex[i] * h_conj / denom;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1693: `let scale = 1.0 / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1702: `solution.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1703: `padded_psf.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1720: `let gcv = n as f64 * rss / (n as f64 - df).powi(2);`
  - **Fix**: Division without zero check - use safe_divide()

### src/denoise.rs

12 issues found:

- Line 126: `median / 0.6745`
  - **Fix**: Division without zero check - use safe_divide()
- Line 141: `ThresholdSelect::Universal => sigma * (2.0 * (n as f64).ln()).sqrt(),`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 141: `ThresholdSelect::Universal => sigma * (2.0 * (n as f64).ln()).sqrt(),`
  - **Fix**: Mathematical operation .ln() without validation
- Line 145: `sigma * (2.0 * (n as f64).ln()).sqrt() * 0.75`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 145: `sigma * (2.0 * (n as f64).ln()).sqrt() * 0.75`
  - **Fix**: Mathematical operation .ln() without validation
- Line 204: `x - (threshold * threshold / x)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 232: `(values[n / 2 - 1] + values[n / 2]) / 2.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 234: `values[n / 2]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 247: `(sorted_deviations[m / 2 - 1] + sorted_deviations[m / 2]) / 2.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 249: `sorted_deviations[m / 2]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 293: `let time: Vec<f64> = (0..n).map(|i| i as f64 / 128.0).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 316: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/denoise_advanced.rs

39 issues found:

- Line 164: `let (denoised_shifted, _) = standard_denoise(&shifted, config, noise_level).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 180: `averaged[i] += result[i] / n_shifts as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 209: `accumulated[(i + shift) % n] += denoised_shifted[i] / n_shifts as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 235: `let shrinkage_factor = signal_var / (signal_var + noise_level * noise_level);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 236: `let threshold = noise_level * (1.0 - shrinkage_factor).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 269: `let n_blocks = (n_coeffs + block_size - 1) / block_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 272: `let base_threshold = noise_level * (2.0 * (signal.len() as f64).ln()).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 272: `let base_threshold = noise_level * (2.0 * (signal.len() as f64).ln()).sqrt();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 273: `let scale_factor = (level_idx + 1) as f64 / config.level as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 288: `block_energy = block_energy.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 293: `let shrinkage = (block_energy - threshold) / block_energy;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 322: `let base_threshold = noise_level * (2.0 * (signal.len() as f64).ln()).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 322: `let base_threshold = noise_level * (2.0 * (signal.len() as f64).ln()).sqrt();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 329: `let scale_factor = (level_idx + 1) as f64 / config.level as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 362: `sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 363: `let median = sorted[sorted.len() / 2];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 369: `deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 370: `let mad = deviations[deviations.len() / 2];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 373: `Ok(mad / 0.6745)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 380: `let mean = detail.iter().sum::<f64>() / detail.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 383: `.sum::<f64>() / detail.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 385: `Ok(variance.sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 391: `detail.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 393: `let q1_idx = detail.len() / 4;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 394: `let q3_idx = 3 * detail.len() / 4;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 398: `Ok(iqr / 1.349)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 407: `let mean = chunk.iter().sum::<f64>() / window as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 410: `.sum::<f64>() / window as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 415: `variances.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 416: `Ok(variances[0].sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 426: `.sum::<f64>() / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 448: `let signal_power = denoised.iter().map(|&x| x * x).sum::<f64>() / denoised.len()...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 449: `let noise_power = noise.iter().map(|&x| x * x).sum::<f64>() / noise.len() as f64...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 452: `Some(10.0 * (signal_power / noise_power).log10())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 470: `let threshold = noise_level * (2.0 * (signal.len() as f64).ln()).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 470: `let threshold = noise_level * (2.0 * (signal.len() as f64).ln()).sqrt();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 497: `.map(|i| (2.0 * PI * i as f64 / n as f64 * 5.0).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 506: `let result = advanced_denoise(&noisy, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 524: `let result = advanced_denoise(&signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/denoise_enhanced.rs

61 issues found:

- Line 199: `let retention_rate = retained_coeffs / total_coeffs as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 270: `averaged[(i + shift) % n] += result.signal[i] / n_shifts as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 277: `let noise_sigma = all_noise_estimates.iter().sum::<f64>() / n_shifts as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 281: `.sum::<f64>() / n_shifts as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 299: `check_finite(&image.as_slice().unwrap(), "image")?;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 330: `all_d_thresholds[0] / (2.0 * (d_detail.len() as f64).ln()).sqrt()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 330: `all_d_thresholds[0] / (2.0 * (d_detail.len() as f64).ln()).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 330: `all_d_thresholds[0] / (2.0 * (d_detail.len() as f64).ln()).sqrt()`
  - **Fix**: Mathematical operation .ln() without validation
- Line 358: `approximations.last().unwrap().clone()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 361: `let noise_sigma = all_d_thresholds[0] / (2.0_f64).powf(levels as f64 / 2.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 361: `let noise_sigma = all_d_thresholds[0] / (2.0_f64).powf(levels as f64 / 2.0);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 363: `threshold_subband(approximations.last().unwrap(), noise_sigma, levels, config)?;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 408: `abs_coeffs.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 411: `(abs_coeffs[abs_coeffs.len() / 2 - 1] + abs_coeffs[abs_coeffs.len() / 2]) / 2.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 413: `abs_coeffs[abs_coeffs.len() / 2]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 442: `ThresholdRule::Universal => noise_sigma * (2.0 * n.ln()).sqrt(),`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 442: `ThresholdRule::Universal => noise_sigma * (2.0 * n.ln()).sqrt(),`
  - **Fix**: Mathematical operation .ln() without validation
- Line 490: `(thresholded, retained as f64 / coeffs.len() as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 506: `(thresholded, retained as f64 / coeffs.len() as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 518: `thresholded[i] = coeff * (1.0 - threshold_sq / coeff_sq);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 523: `(thresholded, retained as f64 / coeffs.len() as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 537: `thresholded[i] = coeff.signum() * (a * abs_coeff - a * threshold) / (a - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 545: `(thresholded, retained as f64 / coeffs.len() as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 560: `let scale = (abs_coeff - threshold) / (upper_threshold - threshold);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 569: `(thresholded, retained as f64 / coeffs.len() as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 581: `thresholded[i] = coeff * (coeff_sq - threshold_sq).sqrt() / coeff.abs();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 581: `thresholded[i] = coeff * (coeff_sq - threshold_sq).sqrt() / coeff.abs();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 586: `(thresholded, retained as f64 / coeffs.len() as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 592: `let n_blocks = (n + block_size - 1) / block_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 605: `block_energy = block_energy.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 617: `let retention_rate = retained_blocks as f64 / n_blocks as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 624: `let max_threshold = noise_sigma * (2.0 * n.ln()).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 624: `let max_threshold = noise_sigma * (2.0 * n.ln()).sqrt();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 631: `let threshold = max_threshold * (i + 1) as f64 / n_candidates as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 666: `let variance = coeffs.iter().map(|&x| x * x).sum::<f64>() / coeffs.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 670: `noise_sigma * noise_sigma / signal_variance.sqrt()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 670: `noise_sigma * noise_sigma / signal_variance.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 679: `let log_n = n.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 684: `noise_sigma * (0.3936 + 0.1829 * log_n).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 697: `abs_coeffs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 703: `let p_value = 2.0 * (1.0 - normal_cdf(abs_val / noise_sigma));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 704: `let fdr_threshold = q * (k + 1) as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 719: `let max_threshold = noise_sigma * (2.0 * (n as f64).ln()).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 719: `let max_threshold = noise_sigma * (2.0 * (n as f64).ln()).sqrt();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 726: `let threshold = max_threshold * (i + 1) as f64 / n_candidates as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 739: `(thresholded[j - 1] + thresholded[j]) / 2.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 768: `let subband_sigma = noise_sigma * 2.0_f64.powf(level as f64 / 2.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 768: `let subband_sigma = noise_sigma * 2.0_f64.powf(level as f64 / 2.0);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 772: `ThresholdRule::Universal => subband_sigma * (2.0 * (flat.len() as f64).ln()).sqr...`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 772: `ThresholdRule::Universal => subband_sigma * (2.0 * (flat.len() as f64).ln()).sqr...`
  - **Fix**: Mathematical operation .ln() without validation
- Line 775: `_ => subband_sigma * (2.0 * (flat.len() as f64).ln()).sqrt(),`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 775: `_ => subband_sigma * (2.0 * (flat.len() as f64).ln()).sqrt(),`
  - **Fix**: Mathematical operation .ln() without validation
- Line 846: `let noise_estimate = (original - denoised).mapv(|x| x * x).sum() / original.len(...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 847: `let signal_power = original.mapv(|x| x * x).sum() / original.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 848: `let snr_improvement = 10.0 * (signal_power / noise_estimate.max(1e-10)).log10();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 852: `/ (2.0 * h_retention.len() as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 858: `/ (3.0 * h_retention.len() as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 869: `0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 869: `0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 885: `let t = 1.0 / (1.0 + p * x);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 910: `let result = denoise_wavelet_1d(&noisy_signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/detrend.rs

22 issues found:

- Line 90: `let mean = x_f64.iter().sum::<f64>() / x_f64.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 99: `let mean_x = x_indices.iter().sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 100: `let mean_y = x_f64.iter().sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 115: `numerator / denominator`
  - **Fix**: Division without zero check - use safe_divide()
- Line 215: `let mean = col.sum() / col.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 220: `let mean = row.sum() / row.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 420: `let factor = a_copy[[j, i]] / a_copy[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 436: `x[i] = (b_copy[i] - sum) / a_copy[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 452: `let detrended = detrend(&signal, Some("constant")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 455: `let mean = detrended.iter().sum::<f64>() / detrended.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 470: `let detrended = detrend(&signal, Some("linear")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 482: `let detrended = detrend(&signal, Some("none")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 501: `let detrended_cols = detrend_axis(&data, Some("linear"), 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 511: `let mean_x = x.iter().sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 512: `let mean_y = col.iter().sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 524: `let slope = numerator / denominator;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 529: `let detrended_rows = detrend_axis(&data, Some("linear"), 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 539: `let mean_x = x.iter().sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 540: `let mean_y = row.iter().sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 552: `let slope = numerator / denominator;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 568: `let detrended = detrend_poly(&signal, 3).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 576: `let detrended_quadratic = detrend_poly(&signal, 2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/dwt/multiscale.rs

1 issues found:

- Line 73: `let max_level = (data_len as f64 / min_length as f64).log2().floor() as usize;`
  - **Fix**: Division without zero check - use safe_divide()

### src/dwt/transform.rs

5 issues found:

- Line 68: `let output_len = (input_len + filter_len - 1) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 99: `let scale_factor = 2.0_f64.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 157: `let scale_factor = 1.0 / 2.0_f64.sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 157: `let scale_factor = 1.0 / 2.0_f64.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 184: `let filter_delay = (filter_len / 2) - 1;`
  - **Fix**: Division without zero check - use safe_divide()

### src/dwt/utils.rs

4 issues found:

- Line 75: `if (sum_lo - 2.0_f64.sqrt()).abs() > tolerance {`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 146: `let omega = pi * k as f64 / points as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 160: `num += (k as f64 / points as f64 / 2.0) * magnitude_squared;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 168: `num / den`
  - **Fix**: Division without zero check - use safe_divide()

### src/dwt2d.rs

11 issues found:

- Line 1078: `x * (1.0 - (threshold * threshold) / (x * x))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1266: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1269: `let decomposition = dwt2d_decompose(&data, Wavelet::Haar, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1278: `let reconstructed = dwt2d_reconstruct(&decomposition, Wavelet::Haar, None).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1296: `let coeffs = wavedec2(&data, Wavelet::Haar, levels, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1302: `let reconstructed = waverec2(&coeffs, Wavelet::Haar, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1319: `let mut decomposition = dwt2d_decompose(&data, Wavelet::Haar, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1392: `let expected_0 = -10.0 * (1.0 - (threshold * threshold) / (10.0 * 10.0));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1407: `let decomposition = dwt2d_decompose(&data, Wavelet::Haar, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1442: `let decomposition = dwt2d_decompose(&data, Wavelet::DB(2), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1451: `let reconstructed = dwt2d_reconstruct(&decomposition, Wavelet::DB(2), None).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/dwt2d_enhanced.rs

12 issues found:

- Line 95: `check_finite(&data.as_slice().unwrap(), "data")?;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 138: `let half_cols = (cols + 1) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 156: `let half_rows = (rows + 1) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 233: `let half_cols = (cols + 1) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 258: `let half_rows = (rows + 1) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 346: `let signal_view = ArrayView2::from_shape((1, len), signal_slice).unwrap();`
  - **Fix**: Handle array creation errors properly
- Line 347: `let lo_view = ArrayView2::from_shape((1, len), lo_slice).unwrap();`
  - **Fix**: Handle array creation errors properly
- Line 348: `let hi_view = ArrayView2::from_shape((1, len), hi_slice).unwrap();`
  - **Fix**: Handle array creation errors properly
- Line 376: `let pad_len = filter_len / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 521: `]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 524: `let result = enhanced_dwt2d_decompose(&data, Wavelet::Haar, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 547: `let result = enhanced_dwt2d_decompose(&data, Wavelet::DB(4), &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/dwt2d_image.rs

25 issues found:

- Line 124: `let sigma = (energy.detail_h + energy.detail_v + energy.detail_d).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 125: `/ ((coeffs[0].detail_h.len() + coeffs[0].detail_v.len() + coeffs[0].detail_d.len...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 127: `.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 130: `let universal_threshold = sigma * (2.0 * n.ln()).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 130: `let universal_threshold = sigma * (2.0 * n.ln()).sqrt();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 135: `let level_factor = 1.0 / (2.0_f64.powi(i as i32));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 153: `let sigma = ((h_sigma + v_sigma + d_sigma) / 3.0).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 153: `let sigma = ((h_sigma + v_sigma + d_sigma) / 3.0).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 157: `/ (level.detail_h.len() + level.detail_v.len() + level.detail_d.len()) as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 161: `let sigma_x = signal_var.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 163: `sigma * sigma / sigma_x`
  - **Fix**: Division without zero check - use safe_divide()
- Line 344: `if level != coeffs.first().unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 377: `1.0 - (compressed_nonzeros as f32 / original_nonzeros as f32)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 420: `let mid = abs_coeffs.len() / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 422: `(abs_coeffs[mid - 1] + abs_coeffs[mid]) / 2.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 433: `let mean = abs_coeffs.iter().sum::<f64>() / abs_coeffs.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 435: `abs_coeffs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / abs_coeffs.len() a...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 490: `x * (1.0 - t_sq / (x * x))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 562: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 572: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 582: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 606: `let edges = detect_edges(&image, Wavelet::Haar, 1, Some(0.5)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 636: `let (compressed_low, ratio_low) = compress_image(&image, Wavelet::DB(2), 2, 0.3)...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 637: `let (compressed_high, ratio_high) = compress_image(&image, Wavelet::DB(2), 2, 0....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 648: `let (no_compression, ratio_zero) = compress_image(&image, Wavelet::DB(2), 2, 0.0...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/dwt2d_validation.rs

30 issues found:

- Line 112: `check_finite(&test_image.as_slice().unwrap(), "test_image")?;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 203: `let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 204: `let rmse = (sum_sq_error / errors.len() as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 204: `let rmse = (sum_sq_error / errors.len() as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 244: `let energy_ratio = output_energy / input_energy;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 248: `approx_percent: 100.0 * approx_energy / output_energy,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 249: `detail_h_percent: 100.0 * h_energy / output_energy,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 250: `detail_v_percent: 100.0 * v_energy / output_energy,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 251: `detail_d_percent: 100.0 * d_energy / output_energy,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 318: `let edge_artifacts = total_artifacts / n_modes;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 319: `let continuity_score = total_continuity / n_modes;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 323: `symmetry_scores.iter().sum::<f64>() / symmetry_scores.len() as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 348: `let standard_time = start.elapsed().as_micros() as f64 / (n_runs as f64 * 1000.0...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 356: `let enhanced_time = start.elapsed().as_micros() as f64 / (n_runs as f64 * 1000.0...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 358: `let speedup = standard_time / enhanced_time.max(0.001);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 401: `let var1 = window1.iter().map(|&x| (x - mu1).powi(2)).sum::<f64>() / (window_siz...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 402: `let var2 = window2.iter().map(|&x| (x - mu2).powi(2)).sum::<f64>() / (window_siz...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 421: `Ok(sum_ssim / n_windows as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 443: `Ok(total_artifacts / perimeter)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 457: `continuity_score += 1.0 / (1.0 + (left_diff - right_diff).abs());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 461: `Ok(continuity_score / n_measurements as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 503: `let preservation_score = 1.0 - (approx_symmetry / (h_symmetry + v_symmetry + 1e-...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 537: `let energy_ratio = total_energy / input_energy;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 546: `let expected_size = ((current_size.0 + 1) / 2, (current_size.1 + 1) / 2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 571: `gradient[[i, j]] = (i + j) as f64 / 128.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 580: `checkerboard[[i, j]] = if (i / 8 + j / 8) % 2 == 0 { 1.0 } else { 0.0 };`
  - **Fix**: Division without zero check - use safe_divide()
- Line 593: `gaussian[[i, j]] = (-(dx*dx + dy*dy) / (2.0 * sigma * sigma)).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 657: `let result = validate_dwt2d(&image, Wavelet::Haar, 1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 666: `let metrics = test_energy_conservation(&image, Wavelet::DB(4)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 677: `assert!(validate_multilevel_dwt2d(&image, Wavelet::Sym(8), 4, 1e-10).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()

### src/emd.rs

42 issues found:

- Line 139: `let energy = imf.iter().map(|&x| x * x).sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 150: `let residue_energy = residue.iter().map(|&x| x * x).sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 151: `let original_energy = signal_f64.iter().map(|&x| x * x).sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 153: `if residue_energy < 1e-10 || residue_energy / original_energy < 1e-2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 233: `mean_env[i] = (upper_env[i] + lower_env[i]) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 279: `sum_squared_diff / sum_squared_prev`
  - **Fix**: Division without zero check - use safe_divide()
- Line 304: `.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 355: `.min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 471: `let slope = (second_val - first_val) / (second_idx - first_idx);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 487: `let slope = (last_val - penultimate_val) / (last_idx - penultimate_idx);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 558: `y_left + (i as f64 - x_left) * (y_right - y_left) / (x_right - x_left);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 588: `* ((values[i + 1] - values[i]) / h[i] - (values[i] - values[i - 1]) / h[i - 1]);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 602: `mu[i] = h[i] / l[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 603: `z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 619: `b[i] = (values[i + 1] - values[i]) / h[i]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 620: `- h[i] * (c_values[i + 1] + 2.0 * c_values[i]) / 3.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 621: `d[i] = (c_values[i + 1] - c_values[i]) / (3.0 * h[i]);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 806: `let energy = avg_imfs.slice(s![i, ..]).map(|&x| x * x).sum() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 864: `let time_points: Vec<f64> = (0..n).map(|i| i as f64 / sample_rate).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 870: `let log_min = min_freq.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 871: `let log_max = max_freq.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 872: `let log_step = (log_max - log_min) / (num_freqs - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 901: `inst_freq.push(first_diff * sample_rate / (2.0 * std::f64::consts::PI));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 915: `inst_freq.push(diff * sample_rate / (4.0 * std::f64::consts::PI));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 921: `inst_freq.push(last_diff * sample_rate / (2.0 * std::f64::consts::PI));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 931: `let log_freq = inst_freq[j].ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 932: `let idx = ((log_freq - log_min) / log_step).round() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 957: `.map(|i| (2.0 * PI * i as f64 / period as f64).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1013: `idx % period == period / 4`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1014: `|| idx % period == period / 4 - 1`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1015: `|| idx % period == period / 4 + 1,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1018: `period / 4`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1025: `idx % period == 3 * period / 4`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1026: `|| idx % period == 3 * period / 4 - 1`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1027: `|| idx % period == 3 * period / 4 + 1,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1030: `3 * period / 4`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1052: `let linear_env = interpolate_envelope(&indices, &values, n, "linear").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1063: `let cubic_env = interpolate_envelope(&indices, &values, n, "cubic").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1076: `.map(|i| (2.0 * PI * i as f64 / 20.0).sin() + 0.5 * (2.0 * PI * i as f64 / 5.0)....`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1082: `let result = emd(&signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1097: `.map(|i| (2.0 * PI * i as f64 / 20.0).sin() + 0.5 * (2.0 * PI * i as f64 / 5.0)....`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1106: `let result = eemd(&signal, &config, ensemble_size, noise_std).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/features/activity.rs

6 issues found:

- Line 43: `let sma = signal_f64.iter().map(|&x| x.abs()).sum::<f64>() / signal_f64.len() as...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 54: `energy_ratio * (1.0 - (*low / bands[bands.len() - 1].1).powi(2)),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 65: `let max_lag = (0.5 * fs).min((signal_f64.len() as f64) / 3.0) as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 89: `let mean = signal.iter().sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 98: `let variance = signal_centered.iter().map(|&x| x * x).sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 115: `sum / (variance * (n - lag) as f64)`
  - **Fix**: Division without zero check - use safe_divide()

### src/features/batch.rs

1 issues found:

- Line 177: `feature_matrix[[0, i]] = *first_features.get(name).unwrap();`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?

### src/features/entropy.rs

19 issues found:

- Line 48: `sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 52: `let bin_width = 2.0 * iqr / (n as f64).powf(1.0 / 3.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 52: `let bin_width = 2.0 * iqr / (n as f64).powf(1.0 / 3.0);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 53: `let num_bins = ((max - min) / bin_width).ceil() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 60: `let bin = ((value - min) / (max - min) * num_bins as f64).floor() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 70: `let probability = count as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 71: `entropy -= probability * probability.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 128: `total += (local_count / (n - m + 1) as f64).ln();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 128: `total += (local_count / (n - m + 1) as f64).ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 131: `total / (n - m + 1) as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 159: `-((count_m_plus_1 / count_m).ln())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 159: `-((count_m_plus_1 / count_m).ln())`
  - **Fix**: Mathematical operation .ln() without validation
- Line 188: `match_count * 2.0 / ((n - m) as f64 * (n - m - 1) as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 216: `idx.sort_by(|&a, &b| pattern[a].partial_cmp(&pattern[b]).unwrap());`
  - **Fix**: Use .get() with proper bounds checking
- Line 221: `.map(|&i| char::from_digit(i as u32, 10).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 233: `let probability = count as f64 / total_patterns;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 234: `entropy -= probability * probability.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 238: `entropy / factorial.ln()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 238: `entropy / factorial.ln()`
  - **Fix**: Mathematical operation .ln() without validation

### src/features/peaks.rs

6 issues found:

- Line 45: `maxima.len() as f64 / minima.len() as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 59: `features.insert("peak_density".to_string(), maxima.len() as f64 / n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 68: `let mean_peak_distance = peak_distances.iter().sum::<f64>() / peak_distances.len...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 78: `/ peak_distances.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 80: `features.insert("peak_distance_std".to_string(), variance.sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 88: `features.insert("crest_factor".to_string(), max_abs / rms);`
  - **Fix**: Division without zero check - use safe_divide()

### src/features/spectral.rs

3 issues found:

- Line 117: `features.insert("low_freq_energy_ratio".to_string(), low_band_energy / total);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 118: `features.insert("mid_freq_energy_ratio".to_string(), mid_band_energy / total);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 121: `high_band_energy / total,`
  - **Fix**: Division without zero check - use safe_divide()

### src/features/statistical.rs

25 issues found:

- Line 14: `let mean = sum / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 19: `let variance = sum_squared_diff / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 20: `let std_dev = variance.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 26: `sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 28: `(sorted[n / 2 - 1] + sorted[n / 2]) / 2.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 30: `sorted[n / 2]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 37: `.min_by(|a, b| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 38: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 41: `.max_by(|a, b| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 42: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 67: `let power = energy / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 72: `let mad: f64 = signal.iter().map(|&x| (x - mean).abs()).sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 77: `features.insert("cv".to_string(), std_dev / mean.abs());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 94: `sum_cubed_diff / ((n - 1.0) * std_dev.powi(3))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 133: `let mean = signal.iter().sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 134: `let variance = signal.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f6...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 136: `variance.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 149: `extract_statistical_features(&signal, &mut features).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 152: `assert_eq!(*features.get("mean").unwrap(), 3.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 153: `assert_eq!(*features.get("median").unwrap(), 3.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 154: `assert_eq!(*features.get("min").unwrap(), 1.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 155: `assert_eq!(*features.get("max").unwrap(), 5.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 156: `assert_eq!(*features.get("range").unwrap(), 4.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 159: `assert!((features.get("variance").unwrap() - 2.0).abs() < 1e-10);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 160: `assert!((features.get("std").unwrap() - 2.0_f64.sqrt()).abs() < 1e-10);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?

### src/features/trend.rs

10 issues found:

- Line 36: `let non_linearity = (r_squared_quad - r_squared) / (1.0 - r_squared);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 59: `let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 60: `let intercept = (sum_y - slope * sum_x) / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 63: `let mean_y = sum_y / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 72: `let r_squared = 1.0 - ss_residual / ss_total;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 113: `/ det;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 119: `/ det;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 125: `/ det;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 128: `let mean_y = sum_y / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 137: `let r_squared = 1.0 - ss_residual / ss_total;`
  - **Fix**: Division without zero check - use safe_divide()

### src/features/zero_crossing.rs

4 issues found:

- Line 26: `zero_crossings as f64 / n as f64,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 32: `let frequency_estimate = (zero_crossings as f64) * fs / (2.0 * n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 37: `let mean = signal.iter().sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 50: `mean_crossings as f64 / n as f64,`
  - **Fix**: Division without zero check - use safe_divide()

### src/filter/analysis.rs

3 issues found:

- Line 122: `.map(|i| i as f64 / (n_points - 1) as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 439: `Ok(peak_freq / bandwidth)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 465: `let t = (target_db - m1) / (m2 - m1);`
  - **Fix**: Division without zero check - use safe_divide()

### src/filter/application.rs

18 issues found:

- Line 115: `let b_norm: Vec<f64> = b.iter().map(|&val| val / a0).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 116: `let a_norm: Vec<f64> = a.iter().map(|&val| val / a0).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 226: `let min_zero = 1.0 / zero.conj();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 240: `gain_adjustment *= -zero.re / min_zero.re;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 257: `let scale = b[0] / min_phase_b[0];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 319: `gd.push(-phase_diff / freq_diff);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 366: `let norm_factor = 1.0 / energy.sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 366: `let norm_factor = 1.0 / energy.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 444: `num_val / den_val`
  - **Fix**: Division without zero check - use safe_divide()
- Line 473: `roots.push(Complex64::new(-trimmed_coeffs[1] / trimmed_coeffs[0], 0.0));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 487: `let sqrt_disc = discriminant.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 488: `roots.push(Complex64::new((-b + sqrt_disc) / (2.0 * a), 0.0));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 489: `roots.push(Complex64::new((-b - sqrt_disc) / (2.0 * a), 0.0));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 491: `let sqrt_disc = (-discriminant).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 492: `roots.push(Complex64::new(-b / (2.0 * a), sqrt_disc / (2.0 * a)));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 493: `roots.push(Complex64::new(-b / (2.0 * a), -sqrt_disc / (2.0 * a)));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 506: `let angle = 2.0 * std::f64::consts::PI * k as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 520: `let correction = p_val / p_prime;`
  - **Fix**: Division without zero check - use safe_divide()

### src/filter/common.rs

4 issues found:

- Line 199: `(PI * digital_freq / 2.0).tan()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 206: `(2.0 + analog_pole) / (2.0 - analog_pole)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 213: `(2.0 + analog_zero) / (2.0 - analog_zero)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 223: `let angle = PI * (2.0 * k as f64 + order as f64 + 1.0) / (2.0 * order as f64);`
  - **Fix**: Division without zero check - use safe_divide()

### src/filter/fir.rs

20 issues found:

- Line 53: `let mid = (numtaps - 1) as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 62: `wc / std::f64::consts::PI`
  - **Fix**: Division without zero check - use safe_divide()
- Line 64: `1.0 - wc / std::f64::consts::PI`
  - **Fix**: Division without zero check - use safe_divide()
- Line 68: `let sinc_val = (wc * std::f64::consts::PI * n).sin() / (std::f64::consts::PI * n...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 73: `if i == numtaps / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 191: `let r = (filter_order + 2) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 200: `let num_bands = bands.len() / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 208: `band_start + (band_end - band_start) * (i as f64) / (band_points as f64 - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 212: `let t = (omega - band_start) / (band_end - band_start);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 229: `extremal_freqs.push(i * (omega_grid.len() - 1) / (r - 1));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 249: `a_matrix[i][r - 1] = if i % 2 == 0 { 1.0 } else { -1.0 } / weight_grid[ext_idx];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 299: `new_extremal.sort_by(|&a, &b| errors[b].partial_cmp(&errors[a]).unwrap());`
  - **Fix**: Use .get() with proper bounds checking
- Line 314: `let n = i as f64 - (numtaps as f64 - 1.0) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 321: `let freq = j as f64 * std::f64::consts::PI / (numtaps as f64 - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 336: `let mid = numtaps / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 338: `let avg = (h[i] + h[numtaps - 1 - i]) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 366: `*w = 0.54 - 0.46 * (2.0 * std::f64::consts::PI * n / (total - 1.0)).cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 373: `*w = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * n / (total - 1.0)).cos());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 380: `let arg = 2.0 * std::f64::consts::PI * n / (total - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 438: `let factor = aug[k][i] / aug[i][i];`
  - **Fix**: Division without zero check - use safe_divide()

### src/filter/iir.rs

50 issues found:

- Line 80: `let hp_poles: Vec<_> = poles.iter().map(|p| warped_freq / p).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 165: `let center_freq = (wl * wh).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 176: `let discriminant = (bandwidth * pole / 2.0).powi(2) + center_freq.powi(2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 177: `let sqrt_disc = discriminant.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 178: `let p1 = bandwidth * pole / 2.0 + sqrt_disc;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 179: `let p2 = bandwidth * pole / 2.0 - sqrt_disc;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 197: `let discriminant = (bandwidth / (2.0 * pole)).powi(2) + center_freq.powi(2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 198: `let sqrt_disc = discriminant.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 199: `let p1 = bandwidth / (2.0 * pole) + sqrt_disc;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 200: `let p2 = bandwidth / (2.0 * pole) - sqrt_disc;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 283: `let epsilon = (10.0_f64.powf(ripple / 10.0) - 1.0).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 283: `let epsilon = (10.0_f64.powf(ripple / 10.0) - 1.0).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 283: `let epsilon = (10.0_f64.powf(ripple / 10.0) - 1.0).sqrt();`
  - **Fix**: Mathematical operation .powf( without validation
- Line 287: `let a = (1.0 / epsilon + (1.0 / epsilon / epsilon + 1.0).sqrt()).ln() / order as...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 287: `let a = (1.0 / epsilon + (1.0 / epsilon / epsilon + 1.0).sqrt()).ln() / order as...`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 287: `let a = (1.0 / epsilon + (1.0 / epsilon / epsilon + 1.0).sqrt()).ln() / order as...`
  - **Fix**: Mathematical operation .ln() without validation
- Line 290: `let theta = std::f64::consts::PI * (2.0 * k as f64 + 1.0) / (2.0 * order as f64)...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 309: `let hp_poles: Vec<_> = poles.iter().map(|p| warped_freq / p).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 383: `let epsilon = 1.0 / (10.0_f64.powf(attenuation / 10.0) - 1.0).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 383: `let epsilon = 1.0 / (10.0_f64.powf(attenuation / 10.0) - 1.0).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 383: `let epsilon = 1.0 / (10.0_f64.powf(attenuation / 10.0) - 1.0).sqrt();`
  - **Fix**: Mathematical operation .powf( without validation
- Line 390: `let a = (epsilon + (epsilon * epsilon + 1.0).sqrt()).ln() / order as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 390: `let a = (epsilon + (epsilon * epsilon + 1.0).sqrt()).ln() / order as f64;`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 390: `let a = (epsilon + (epsilon * epsilon + 1.0).sqrt()).ln() / order as f64;`
  - **Fix**: Mathematical operation .ln() without validation
- Line 394: `let theta = std::f64::consts::PI * (2.0 * k as f64 + 1.0) / (2.0 * order as f64)...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 402: `let inv_pole = 1.0 / pole;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 408: `let theta = std::f64::consts::PI * (2.0 * k as f64 + 1.0) / (2.0 * order as f64)...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 409: `let zero_imag = 1.0 / theta.cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 423: `let hp_poles: Vec<_> = poles.iter().map(|p| warped_freq / p).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 424: `let hp_zeros: Vec<_> = zeros.iter().map(|z| warped_freq / z).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 514: `let epsilon_p = (10.0_f64.powf(passband_ripple / 10.0) - 1.0).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 514: `let epsilon_p = (10.0_f64.powf(passband_ripple / 10.0) - 1.0).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 514: `let epsilon_p = (10.0_f64.powf(passband_ripple / 10.0) - 1.0).sqrt();`
  - **Fix**: Mathematical operation .powf( without validation
- Line 515: `let epsilon_s = (10.0_f64.powf(stopband_attenuation / 10.0) - 1.0).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 515: `let epsilon_s = (10.0_f64.powf(stopband_attenuation / 10.0) - 1.0).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 515: `let epsilon_s = (10.0_f64.powf(stopband_attenuation / 10.0) - 1.0).sqrt();`
  - **Fix**: Mathematical operation .powf( without validation
- Line 532: `let a = (1.0 / epsilon_p).asinh() / order as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 535: `let theta = std::f64::consts::PI * (2.0 * k as f64 + 1.0) / (2.0 * order as f64)...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 542: `let mod_factor = 1.0 + (epsilon_s / epsilon_p).ln() / (2.0 * order as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 542: `let mod_factor = 1.0 + (epsilon_s / epsilon_p).ln() / (2.0 * order as f64);`
  - **Fix**: Mathematical operation .ln() without validation
- Line 547: `if k < order / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 548: `let zero_freq = 1.5 + 0.5 * k as f64 / (order as f64 / 2.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 550: `if order % 2 == 0 || k < order / 2 - 1 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 566: `let hp_poles: Vec<_> = poles.iter().map(|p| warped_freq / p).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 567: `let hp_zeros: Vec<_> = zeros.iter().map(|z| warped_freq / z).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 698: `let theta = std::f64::consts::PI * (2.0 * k as f64 + 1.0) / (2.0 * order as f64)...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 699: `let radius = 1.0 - 0.1 * (order as f64 - 8.0).min(5.0) / 10.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 720: `let hp_poles: Vec<_> = bessel_poles.iter().map(|p| warped_freq / p).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 841: `let b_normalized: Vec<f64> = b.iter().map(|&coeff| coeff / a0).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 842: `let a_normalized: Vec<f64> = a.iter().map(|&coeff| coeff / a0).collect();`
  - **Fix**: Division without zero check - use safe_divide()

### src/filter/mod.rs

13 issues found:

- Line 122: `let (_b, a) = result.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 133: `let h = result.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 137: `for i in 0..h.len() / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 145: `let (b, a) = butter(4, 0.2, "lowpass").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 149: `let analysis = result.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 158: `let (b, a) = butter(2, 0.3, "lowpass").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 163: `let filtered = result.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 173: `let (b, a) = result.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 182: `let (_b, a) = butter(4, 0.2, "lowpass").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 186: `let stability = result.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 203: `let (_b, a) = result.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 212: `assert_eq!(filter_type.unwrap(), FilterType::Lowpass);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 216: `assert_eq!(filter_type.unwrap(), FilterType::Highpass);`
  - **Fix**: Replace with ? operator or .ok_or()

### src/filter/parallel.rs

10 issues found:

- Line 133: `((n / n_cores).max(filter_len * 4)).min(8192)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 140: `let n_chunks = (n + chunk - overlap - 1) / (chunk - overlap);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 200: `y[i] += b[j] * x[i - j] / a0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 205: `y[i] -= a[j] * y[i - j] / a0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 231: `let n_chunks = (na + chunk - overlap - 1) / (chunk - overlap);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 275: `let start = (nv - 1) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 326: `let start = (nv - 1) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 397: `"same" => ker_rows / 2,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 404: `"same" => ker_cols / 2,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 537: `data.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()

### src/filter/specialized.rs

21 issues found:

- Line 50: `let r = 1.0 - std::f64::consts::PI * notch_freq / quality_factor;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 175: `let zero = 1.0 / pole.conj();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 218: `let zero1 = 1.0 / pole1.conj();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 219: `let zero2 = 1.0 / pole2.conj();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 264: `let center = num_taps / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 273: `*item = 2.0 / (std::f64::consts::PI * n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 284: `0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (num_taps - 1) as f64).co...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 327: `let center = num_taps / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 334: `*item = (-1.0_f64).powi(n + 1) / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 341: `0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (num_taps - 1) as f64).co...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 381: `let normalized_h: Vec<f64> = h.iter().map(|&x| x / num_taps as f64).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 422: `let center = (num_taps - 1) as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 433: `*item = arg.sin() / arg;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 440: `0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (num_taps - 1) as f64).co...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 519: `let a_gain = 10.0_f64.powf(gain_db / 40.0); // Convert dB to linear`
  - **Fix**: Mathematical operation .powf( without validation
- Line 520: `let q = 1.0 / (2.0 * (bandwidth * std::f64::consts::LN_2 / 2.0).sinh());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 522: `let alpha = omega.sin() / (2.0 * q);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 529: `let a0 = 1.0 + alpha / a_gain;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 531: `let a2 = 1.0 - alpha / a_gain;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 534: `let b = vec![b0 / a0, b1 / a0, b2 / a0];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 535: `let a = vec![1.0, a1 / a0, a2 / a0];`
  - **Fix**: Division without zero check - use safe_divide()

### src/filter/transform.rs

50 issues found:

- Line 53: `let fs_2 = sample_rate / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 59: `let digital_zero = (fs_2 + zero) / (fs_2 - zero);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 65: `let digital_pole = (fs_2 + pole) / (fs_2 - pole);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 187: `let b_normalized: Vec<f64> = b.iter().map(|&coeff| coeff / a0).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 188: `let a_normalized: Vec<f64> = a.iter().map(|&coeff| coeff / a0).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 235: `let gain = if b.is_empty() { 0.0 } else { b[0] / a[0] };`
  - **Fix**: Division without zero check - use safe_divide()
- Line 314: `transformed_zeros.push(wc / zero);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 327: `let transformed_poles: Vec<_> = poles.iter().map(|&p| wc / p).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 370: `let wc = (wl * wh).sqrt(); // Center frequency`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 380: `let discriminant = (bw * zero / 2.0).powi(2) + wc.powi(2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 382: `let sqrt_disc = discriminant.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 383: `let z1 = bw * zero / 2.0 + sqrt_disc;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 384: `let z2 = bw * zero / 2.0 - sqrt_disc;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 388: `let sqrt_disc = (-discriminant).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 389: `let z1 = Complex64::new((bw * zero / 2.0).re, sqrt_disc.re);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 390: `let z2 = Complex64::new((bw * zero / 2.0).re, -sqrt_disc.re);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 399: `let discriminant = (bw * pole / 2.0).powi(2) + wc.powi(2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 401: `let sqrt_disc = discriminant.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 402: `let p1 = bw * pole / 2.0 + sqrt_disc;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 403: `let p2 = bw * pole / 2.0 - sqrt_disc;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 407: `let sqrt_disc = (-discriminant).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 408: `let p1 = Complex64::new((bw * pole / 2.0).re, sqrt_disc.re);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 409: `let p2 = Complex64::new((bw * pole / 2.0).re, -sqrt_disc.re);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 458: `let wc = (wl * wh).sqrt(); // Center frequency`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 468: `let discriminant = (bw / (2.0 * zero)).powi(2) + wc.powi(2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 470: `let sqrt_disc = discriminant.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 471: `let z1 = bw / (2.0 * zero) + sqrt_disc;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 472: `let z2 = bw / (2.0 * zero) - sqrt_disc;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 476: `let sqrt_disc = (-discriminant).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 477: `let z1 = Complex64::new((bw / (2.0 * zero)).re, sqrt_disc.re);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 478: `let z2 = Complex64::new((bw / (2.0 * zero)).re, -sqrt_disc.re);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 487: `let discriminant = (bw / (2.0 * pole)).powi(2) + wc.powi(2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 489: `let sqrt_disc = discriminant.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 490: `let p1 = bw / (2.0 * pole) + sqrt_disc;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 491: `let p2 = bw / (2.0 * pole) - sqrt_disc;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 495: `let sqrt_disc = (-discriminant).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 496: `let p1 = Complex64::new((bw / (2.0 * pole)).re, sqrt_disc.re);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 497: `let p2 = Complex64::new((bw / (2.0 * pole)).re, -sqrt_disc.re);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 505: `for _ in 0..num_added_zeros / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 537: `let b_norm: Vec<f64> = b.iter().map(|&coeff| coeff / a0).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 538: `let a_norm: Vec<f64> = a.iter().map(|&coeff| coeff / a0).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 565: `roots.push(Complex64::new(-trimmed_coeffs[1] / trimmed_coeffs[0], 0.0));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 579: `let sqrt_disc = discriminant.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 580: `roots.push(Complex64::new((-b + sqrt_disc) / (2.0 * a), 0.0));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 581: `roots.push(Complex64::new((-b - sqrt_disc) / (2.0 * a), 0.0));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 583: `let sqrt_disc = (-discriminant).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 584: `roots.push(Complex64::new(-b / (2.0 * a), sqrt_disc / (2.0 * a)));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 585: `roots.push(Complex64::new(-b / (2.0 * a), -sqrt_disc / (2.0 * a)));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 598: `let angle = 2.0 * std::f64::consts::PI * k as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 610: `let correction = p_val / p_prime;`
  - **Fix**: Division without zero check - use safe_divide()

### src/filter_banks.rs

48 issues found:

- Line 209: `PI * (k as f64 + 0.5) * (n as f64 - filter_length as f64 / 2.0 + 0.5)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 210: `/ num_channels as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 224: `let phase = 2.0 * PI * k as f64 * n as f64 / num_channels as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 255: `PI * k as f64 * (2.0 * n as f64 + 1.0) / (2.0 * num_channels as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 262: `2.0 * analysis_filters[[k, n]] / num_channels as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 277: `if k < num_channels / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 361: `input.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 362: `filter.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 397: `upsampled.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 398: `filter.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 523: `let passband_end = magnitude_responses.ncols() / (2 * self.num_channels);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 545: `let stopband_start = magnitude_responses.ncols() / self.num_channels;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 727: `input.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 728: `filter.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 763: `upsampled.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 764: `filter.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 823: `let arg = 2.0 * (n as f64) / (filter_length - 1) as f64 - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 825: `let i0_arg = Self::modified_bessel_i0(beta * (1.0 - arg * arg).sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 826: `prototype[n] = i0_arg / i0_beta;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 833: `0.54 - 0.46 * (2.0 * PI * n as f64 / (filter_length - 1) as f64).cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 840: `0.5 * (1.0 - (2.0 * PI * n as f64 / (filter_length - 1) as f64).cos());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 846: `let arg = 2.0 * PI * n as f64 / (filter_length - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 868: `let t = x / 3.75;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 877: `let t = 3.75 / x.abs();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 886: `/ x.abs().sqrt()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 886: `/ x.abs().sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 955: `pole * (0.99 / pole.norm())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 988: `stabilized_a[i] *= 0.95 / stabilized_a[i].abs();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1014: `roots.push(Complex64::new(-coeffs[1] / coeffs[0], 0.0));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1025: `let sqrt_disc = discriminant.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1026: `roots.push(Complex64::new((-b + sqrt_disc) / (2.0 * a), 0.0));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1027: `roots.push(Complex64::new((-b - sqrt_disc) / (2.0 * a), 0.0));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1029: `let sqrt_disc = (-discriminant).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1030: `roots.push(Complex64::new(-b / (2.0 * a), sqrt_disc / (2.0 * a)));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1031: `roots.push(Complex64::new(-b / (2.0 * a), -sqrt_disc / (2.0 * a)));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1077: `let qmf = QmfBank::new(4, FilterBankType::Orthogonal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1085: `let qmf = QmfBank::new(2, FilterBankType::PerfectReconstruction).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1089: `let subbands = qmf.analysis(&input).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1093: `let reconstructed = qmf.synthesis(&subbands).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1102: `let wavelet_bank = WaveletFilterBank::new("db4", 2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1106: `let coeffs = wavelet_bank.decompose(&input).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1110: `let reconstructed = wavelet_bank.reconstruct(&coeffs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1118: `let cmfb = CosineModulatedFilterBank::new(4, 2, FilterBankWindow::Hann).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1121: `let subbands = cmfb.analysis(&input).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1124: `let reconstructed = cmfb.synthesis(&subbands).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1130: `let qmf = QmfBank::new(2, FilterBankType::Orthogonal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1131: `let analysis = qmf.analyze_properties().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1144: `IirStabilizer::stabilize_filter(&b, &a, StabilizationMethod::RadialProjection).u...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/higher_order.rs

32 issues found:

- Line 218: `let i_idx = (f1 * nfft as f64 / fs).round() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 222: `let j_idx = (f2 * nfft as f64 / fs).round() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 226: `let sum_idx = (sum_freq * nfft as f64 / fs).round() as usize % nfft;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 236: `.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 240: `bicoherence[[i, j]] = bis_complex[[i, j]].norm() / norm_factor;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 272: `let _freq_step = config.fs / nfft as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 273: `let max_freq = config.fs / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 274: `let freq_bins = (nfft / 2) + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 305: `let n_bins = (nfft / 2) + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 347: `let n_bins = (nfft / 2) + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 385: `((n - overlap_samples) as f64 / step as f64).floor() as usize`
  - **Fix**: Division without zero check - use safe_divide()
- Line 395: `let n_bins = (nfft / 2) + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 427: `bispectrum_avg.mapv_inplace(|x| x / n_segments as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 485: `let _freq_step = config.fs / nfft as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 486: `let max_freq = config.fs / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 487: `let freq_bins = (nfft / 2) + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 497: `power_spectrum[i] = fft_result[i].norm_sqr() / nfft as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 515: `let mean = signal.sum() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 519: `let max_lag = size.min(n / 3);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 553: `triple_corr.mapv_inplace(|x| x / n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 594: `let mut result = Array2::zeros((nfft / 2 + 1, nfft / 2 + 1));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 595: `for j in 0..(nfft / 2 + 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 611: `for i in 0..(nfft / 2 + 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 666: `let n_bins = (nfft / 2) + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 723: `let n_bins = (nfft / 2) + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 726: `let _freq_step = fs / nfft as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 727: `let max_freq = fs / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 776: `let max_bandwidth = n_bins / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 798: `cumulative[i] = sum / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 839: `let n_bins = (nfft / 2) + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 847: `skewness[i] = bis_complex[[i, i]].norm() / power_spectrum[i].powf(1.5);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 847: `skewness[i] = bis_complex[[i, i]].norm() / power_spectrum[i].powf(1.5);`
  - **Fix**: Mathematical operation .powf( without validation

### src/hilbert.rs

32 issues found:

- Line 113: `h.iter_mut().take(n / 2).skip(1).for_each(|val| {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 118: `h.iter_mut().skip(n / 2 + 1).for_each(|val| {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 158: `let scale = 1.0 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 215: `.map(|c| (c.re.powi(2) + c.im.powi(2)).sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 298: `unwrapped_phase.push(unwrapped_phase.last().unwrap() + diff);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 306: `inst_freq.push(fs * (unwrapped_phase[1] - unwrapped_phase[0]) / (2.0 * PI));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 310: `let freq = fs * (unwrapped_phase[i + 1] - unwrapped_phase[i - 1]) / (4.0 * PI);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 316: `inst_freq.push(fs * (unwrapped_phase[last_idx] - unwrapped_phase[last_idx - 1]) ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 404: `unwrapped_phase.push(unwrapped_phase.last().unwrap() + diff);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 423: `let dt = 1.0 / sample_rate;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 430: `let analytic = hilbert(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 435: `let start_idx = n / 4;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 436: `let end_idx = 3 * n / 4;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 439: `let magnitude = (analytic[i].re.powi(2) + analytic[i].im.powi(2)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 469: `let dt = 1.0 / fs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 482: `let envelope_result = envelope(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 507: `for (i, &ti) in t.iter().enumerate().skip(n / 10).take(8 * n / 10) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 523: `/ max_points.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 525: `/ min_points.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 539: `let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 542: `let duration = (n - 1) as f64 / fs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 548: `let _freq = f0 + (f1 - f0) * ti / duration;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 549: `let phase = 2.0 * PI * (f0 * ti + 0.5 * (f1 - f0) * ti.powi(2) / duration);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 555: `let inst_freq = instantaneous_frequency(&signal, fs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 559: `let start_idx = n / 5;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 560: `let end_idx = 4 * n / 5;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 564: `let expected_freq = f0 + (f1 - f0) * ti / duration;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 580: `let dt = 1.0 / fs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 587: `let phase = instantaneous_phase(&signal, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 593: `let start_idx = n / 5;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 594: `let end_idx = 4 * n / 5;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 597: `let phase_rate = (phase[i] - phase[i - 1]) / dt;`
  - **Fix**: Division without zero check - use safe_divide()

### src/hr_spectral.rs

27 issues found:

- Line 113: `eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 183: `eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 220: `.map(|&z| z.arg() / (2.0 * PI))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 229: `source_freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 334: `.min_by(|a, b| a.1.norm().partial_cmp(&b.1.norm()).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 335: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 348: `.map(|z| z.arg() / (2.0 * PI))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 357: `source_freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 392: `if order >= n / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 426: `let freq = root.arg() / (2.0 * PI);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 432: `source_freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 463: `correlation[[i, j]] = sum / n_snapshots as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 495: `if eigenval / max_eigenvalue < config.eigenvalue_threshold {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 501: `Ok(eigenvalues.len() / 2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 538: `1.0 / projection_norm`
  - **Fix**: Division without zero check - use safe_divide()
- Line 572: `1.0 / quadratic_form.re`
  - **Fix**: Division without zero check - use safe_divide()
- Line 612: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 614: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 647: `autocorr[[i, j]] = Complex64::new(sum / (n - lag) as f64, 0.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 676: `let normalized_coeffs: Vec<Complex64> = coeffs.iter().map(|&c| c / max_coeff).co...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 685: `let first_idx = first_nonzero.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 694: `return Ok(vec![-effective_coeffs[1] / effective_coeffs[0]]);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 716: `companion[[i, effective_n - 1]] = -effective_coeffs[effective_n - i] / leading_c...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 760: `let result = music(&data, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 782: `let result = minimum_variance(&data, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 802: `let result = pisarenko(&data, 2, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 836: `let autocorr = create_autocorrelation_matrix(&data, 3).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/image_features/color.rs

27 issues found:

- Line 34: `let r_mean = r_sum / n_pixels as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 35: `let g_mean = g_sum / n_pixels as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 36: `let b_mean = b_sum / n_pixels as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 42: `r_mean / g_mean`
  - **Fix**: Division without zero check - use safe_divide()
- Line 50: `r_mean / b_mean`
  - **Fix**: Division without zero check - use safe_divide()
- Line 58: `g_mean / b_mean`
  - **Fix**: Division without zero check - use safe_divide()
- Line 77: `let r_std = (r_var_sum / n_pixels as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 77: `let r_std = (r_var_sum / n_pixels as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 78: `let g_std = (g_var_sum / n_pixels as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 78: `let g_std = (g_var_sum / n_pixels as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 79: `let b_std = (b_var_sum / n_pixels as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 79: `let b_std = (b_var_sum / n_pixels as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 86: `features.insert("color_homogeneity".to_string(), min_std / max_std);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 106: `((image[[i, j, 0]].into() + image[[i, j, 1]].into()) / 2.0 - image[[i, j, 2]].in...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 109: `let rg_mean = rg_diff.iter().sum::<f64>() / n_pixels as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 111: `(rg_diff.iter().map(|&x| (x - rg_mean).powi(2)).sum::<f64>() / n_pixels as f64)....`
  - **Fix**: Division without zero check - use safe_divide()
- Line 111: `(rg_diff.iter().map(|&x| (x - rg_mean).powi(2)).sum::<f64>() / n_pixels as f64)....`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 113: `let yb_mean = yb_diff.iter().sum::<f64>() / n_pixels as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 115: `(yb_diff.iter().map(|&x| (x - yb_mean).powi(2)).sum::<f64>() / n_pixels as f64)....`
  - **Fix**: Division without zero check - use safe_divide()
- Line 115: `(yb_diff.iter().map(|&x| (x - yb_mean).powi(2)).sum::<f64>() / n_pixels as f64)....`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 118: `(rg_std.powi(2) + yb_std.powi(2)).sqrt() + 0.3 * (rg_mean.powi(2) + yb_mean.powi...`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 140: `hue = 60.0 * ((g - b) / (max - min)) % 360.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 142: `hue = 60.0 * ((b - r) / (max - min) + 2.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 144: `hue = 60.0 * ((r - g) / (max - min) + 4.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 152: `let bin = (hue / 20.0).floor() as usize % 18;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 160: `.map(|&count| count as f64 / n_pixels as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 171: `.map(|&p| -p * p.ln())`
  - **Fix**: Mathematical operation .ln() without validation

### src/image_features/edge.rs

8 issues found:

- Line 46: `let magnitude = (gx * gx + gy * gy).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 64: `let mean_gradient = edge_sum / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 68: `let edge_percentage = edge_count as f64 / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 76: `/ n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 77: `let std_dev = variance.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 84: `let bin = (((dir + std::f64::consts::PI) / (2.0 * std::f64::consts::PI)) * 8.0)....`
  - **Fix**: Division without zero check - use safe_divide()
- Line 93: `.map(|&count| count as f64 / n)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 105: `.map(|&p| -p * p.ln())`
  - **Fix**: Mathematical operation .ln() without validation

### src/image_features/haralick.rs

8 issues found:

- Line 21: `glcm.mapv(|x| x / sum)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 63: `std_i = std_i.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 64: `std_j = std_j.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 72: `(i as f64 - mean_i) * (j as f64 - mean_j) * norm_glcm[[i, j]] / (std_i * std_j);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 91: `idm += norm_glcm[[i, j]] / (1.0 + (i as isize - j as isize).pow(2) as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 115: `.map(|&p| -p * p.ln())`
  - **Fix**: Mathematical operation .ln() without validation
- Line 123: `.map(|&p| -p * p.ln())`
  - **Fix**: Mathematical operation .ln() without validation
- Line 152: `.map(|&p| -p * p.ln())`
  - **Fix**: Mathematical operation .ln() without validation

### src/image_features/histogram.rs

6 issues found:

- Line 29: `let bin_width = (max_val - min_val) / bin_count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 32: `let bin = ((val - min_val) / bin_width)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 44: `.map(|&count| count as f64 / total)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 67: `let mode_value = min_val + (mode_bin as f64 + 0.5) * (max_val - min_val) / bin_c...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 74: `.map(|&p| -p * p.ln())`
  - **Fix**: Mathematical operation .ln() without validation
- Line 87: `used_bins as f64 / bin_count as f64,`
  - **Fix**: Division without zero check - use safe_divide()

### src/image_features/lbp.rs

7 issues found:

- Line 61: `let lbp_hist_norm: Vec<f64> = lbp_hist.iter().map(|&count| count as f64 / total)...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 99: `uniform_count as f64 / total,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 109: `entropy -= p * p.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 147: `features.insert("lbp_spots".to_string(), spots as f64 / total);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 148: `features.insert("lbp_flat".to_string(), flat as f64 / total);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 149: `features.insert("lbp_edges".to_string(), edges as f64 / total);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 150: `features.insert("lbp_corners".to_string(), corners as f64 / total);`
  - **Fix**: Division without zero check - use safe_divide()

### src/image_features/moments.rs

13 issues found:

- Line 36: `let x_centroid = m10 / m00;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 37: `let y_centroid = m01 / m00;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 60: `let norm_factor = mu00.powf(1.0 + 1.0); // 1.0 + 1.0 is exponent for 2nd order m...`
  - **Fix**: Mathematical operation .powf( without validation
- Line 61: `let eta11 = mu11 / norm_factor;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 62: `let eta20 = mu20 / norm_factor;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 63: `let eta02 = mu02 / norm_factor;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 81: `let orientation = 0.5 * (2.0 * mu11 / (mu20 - mu02)).atan();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 86: `let major_axis = 2.0 * ((mu20 + mu02 + common.sqrt()) / mu00).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 86: `let major_axis = 2.0 * ((mu20 + mu02 + common.sqrt()) / mu00).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 87: `let minor_axis = 2.0 * ((mu20 + mu02 - common.sqrt()) / mu00).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 87: `let minor_axis = 2.0 * ((mu20 + mu02 - common.sqrt()) / mu00).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 90: `let eccentricity = (1.0 - (minor_axis / major_axis).powi(2)).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 90: `let eccentricity = (1.0 - (minor_axis / major_axis).powi(2)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation

### src/image_features/statistical.rs

10 issues found:

- Line 19: `let mean = sum / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 24: `let variance = sum_squared_diff / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 25: `let std_dev = variance.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 33: `(sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 35: `sorted[sorted.len() / 2]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 43: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 47: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 62: `let rms = (energy / n).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 62: `let rms = (energy / n).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 68: `features.insert("intensity_cv".to_string(), std_dev / mean.abs());`
  - **Fix**: Division without zero check - use safe_divide()

### src/image_features/texture.rs

12 issues found:

- Line 38: `/ 8.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 46: `/ 8.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 56: `gradient_mag[[i, j]] = (gx * gx + gy * gy).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 65: `let mean = flat_gradient.iter().sum::<f64>() / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 70: `/ n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 71: `let std_dev = variance.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 78: `let coarseness = 1.0 / mean;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 85: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 89: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 90: `let contrast = (max_val - min_val) / (max_val + min_val + 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 94: `let energy = flat_gradient.iter().map(|&x| x * x).sum::<f64>() / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 98: `let directionality = variance / (mean * mean + 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()

### src/image_features/utils.rs

3 issues found:

- Line 14: `sum_cubed_diff / ((n - 1.0) * std_dev.powi(3))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 72: `row = ((image[[i, j]] - min_val) / (max_val - min_val) * (num_levels - 1) as f64...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 74: `col = ((image[[i, j + distance]] - min_val) / (max_val - min_val)`
  - **Fix**: Division without zero check - use safe_divide()

### src/interpolate/advanced.rs

17 issues found:

- Line 79: `kernel_sigma * (-0.5 * (x1 - x2).powi(2) / (kernel_length * kernel_length)).exp(...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 531: `let h_norm = h / range;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 551: `nugget + (sill - nugget) * (1.0 - (-3.0 * h / range).exp())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 570: `nugget + (sill - nugget) * (1.0 - (-9.0 * h * h / (range * range)).exp())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 614: `move |r: f64| (1.0 + epsilon * r * r).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 625: `move |r: f64| 1.0 / (1.0 + epsilon * r * r).sqrt()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 625: `move |r: f64| 1.0 / (1.0 + epsilon * r * r).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 637: `r * r * r.ln()`
  - **Fix**: Mathematical operation .ln() without validation
- Line 653: `let result = gaussian_process_interpolate(&signal, 2.0, 1.0, 0.01).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 668: `let variogram = |h: f64| 1.0 - (-h / 2.0).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 670: `let result = kriging_interpolate(&signal, variogram, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 683: `let result = rbf_interpolate(&signal, rbf, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 695: `let result = minimum_energy_interpolate(&signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 762: `let result1 = gaussian_process_interpolate(&signal, 1.0, 1.0, 0.01).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 763: `let result2 = kriging_interpolate(&signal, |_| 1.0, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 764: `let result3 = rbf_interpolate(&signal, |_| 1.0, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 765: `let result4 = minimum_energy_interpolate(&signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/interpolate/basic.rs

14 issues found:

- Line 99: `result[i] = y1 + (y2 - y1) * (x - x1) / (x2 - x1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 193: `let result = linear_interpolate(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 200: `let result = linear_interpolate(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 209: `let result = linear_interpolate(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 219: `let result = linear_interpolate(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 229: `let result = linear_interpolate(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 245: `let result = nearest_neighbor_interpolate(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 252: `let result = nearest_neighbor_interpolate(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 262: `let result = nearest_neighbor_interpolate(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 272: `let result = nearest_neighbor_interpolate(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 290: `let result = linear_interpolate(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 293: `let result = nearest_neighbor_interpolate(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 300: `let result = linear_interpolate(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 303: `let result = nearest_neighbor_interpolate(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/interpolate/core.rs

6 issues found:

- Line 115: `let variogram = |h: f64| -> f64 { 1.0 - (-h / 10.0).exp() };`
  - **Fix**: Division without zero check - use safe_divide()
- Line 121: `let rbf = |r: f64| -> f64 { (-r * r / (2.0 * 10.0 * 10.0)).exp() };`
  - **Fix**: Division without zero check - use safe_divide()
- Line 246: `((i as f64 - vi as f64).powi(2) + (j as f64 - vj as f64).powi(2)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 278: `let half_window = window_size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 297: `result[i] = sum / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 482: `let result = nearest_neighbor_interpolate_2d(&image).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/interpolate/mod.rs

10 issues found:

- Line 366: `let result1 = linear(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 369: `let result2 = cubic_spline(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 372: `let (result3, _method) = auto(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 385: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 426: `let result1 = interpolate(&signal, InterpolationMethod::Linear, &config).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 427: `let result2 = linear_interpolate(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 428: `let result3 = cubic_spline_interpolate(&signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 429: `let result4 = sinc_interpolate(&signal, 0.4).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 430: `let (result5, _) = auto_interpolate(&signal, &config, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 459: `let coeffs = polynomial_fit(&x, &y, 2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/interpolate/spectral.rs

35 issues found:

- Line 92: `x.sin() / x`
  - **Fix**: Division without zero check - use safe_divide()
- Line 97: `let x = PI * distance / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 101: `x.sin() / x`
  - **Fix**: Division without zero check - use safe_divide()
- Line 114: `result[missing_idx] = sum / weight_sum;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 216: `let scale = 1.0 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 233: `let diff = (&result - &prev_result).mapv(|x| x.powi(2)).sum().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 234: `let norm = result.mapv(|x| x.powi(2)).sum().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 236: `if diff / norm < config.convergence_threshold {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 313: `let fold_size = n_valid / k;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 348: `total_error += fold_error / (end - start) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 351: `let avg_error = total_error / k as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 455: `let ratio = input_length as f64 / target_length as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 459: `let kernel_half = kernel.len() / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 488: `sum / weight_sum`
  - **Fix**: Division without zero check - use safe_divide()
- Line 500: `let half_length = config.kernel_length / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 503: `let x = (i as f64 - half_length as f64) / config.oversampling_factor as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 510: `pi_x.sin() / pi_x`
  - **Fix**: Division without zero check - use safe_divide()
- Line 524: `let idx = position.round() as i32 + kernel.len() as i32 / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 535: `let alpha = (length - 1) as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 536: `let x = (n as f64 - alpha) / alpha;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 539: `bessel_i0(beta * (1.0 - x * x).sqrt()) / bessel_i0(beta)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 539: `bessel_i0(beta * (1.0 - x * x).sqrt()) / bessel_i0(beta)`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 549: `let x_half_squared = (x / 2.0).powi(2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 552: `term *= x_half_squared / (k as f64).powi(2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 627: `product *= (x - x_known[j]) / denominator;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 764: `dd_table[i][j] = (dd_table[i + 1][j - 1] - dd_table[i][j - 1]) / denominator;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 797: `let result = sinc_interpolate(&signal, 0.4).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 820: `let result = spectral_interpolate(&signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 836: `let (result, method) = auto_interpolate(&signal, &config, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 859: `let (result, _method) = auto_interpolate(&signal, &config, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 870: `let result1 = sinc_interpolate(&signal, 0.4).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 871: `let result2 = spectral_interpolate(&signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 872: `let (result3, _) = auto_interpolate(&signal, &config, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 904: `let result = lagrange_interpolate(&x_known, &y_known, &x_target).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 918: `let coeffs = polynomial_fit(&x, &y, 2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/interpolate/spline.rs

20 issues found:

- Line 101: `rhs[i] = 3.0 * ((y[i + 1] - y[i]) / h_i1 - (y[i] - y[i - 1]) / h_i);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 162: `let t_norm = (t - x1) / h;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 169: `* ((1.0 - t_norm) * h * h * d1 / 6.0 + t_norm * h * h * d2 / 6.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 269: `let delta1 = (y[i] - y[i - 1]) / h1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 270: `let delta2 = (y[i + 1] - y[i]) / h2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 279: `slopes[i] = (w1 + w2) / (w1 / delta1 + w2 / delta2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 286: `let delta1 = (y[1] - y[0]) / h1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 287: `let delta2 = (y[n_valid - 1] - y[n_valid - 2]) / h2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 336: `let t_norm = (t - x1) / h;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 370: `let result = cubic_spline_interpolate(&signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 378: `let result = cubic_spline_interpolate(&signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 395: `let result = cubic_spline_interpolate(&signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 415: `let result = cubic_hermite_interpolate(&signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 423: `let result = cubic_hermite_interpolate(&signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 438: `let result = cubic_hermite_interpolate(&signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 460: `let result = cubic_spline_interpolate(&signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 463: `let result = cubic_hermite_interpolate(&signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 474: `let result = cubic_spline_interpolate(&signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 477: `let result = cubic_hermite_interpolate(&signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 487: `let result = cubic_spline_interpolate(&signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/kalman.rs

30 issues found:

- Line 213: `let innovation_mean = innovation_sum / (innovation_history.len() as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 222: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 226: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 242: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 246: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 454: `let gamma = (n_states as f64 + lambda).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 457: `let w_m = vec![lambda / (n_states as f64 + lambda)];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 458: `let mut w_m_i = vec![1.0 / (2.0 * (n_states as f64 + lambda)); 2 * n_states];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 462: `let w_c = vec![lambda / (n_states as f64 + lambda) + (1.0 - alpha * alpha + beta...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 463: `let mut w_c_i = vec![1.0 / (2.0 * (n_states as f64 + lambda)); 2 * n_states];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 490: `let diff_col = diff.clone().into_shape_with_order((diff.len(), 1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 491: `let diff_row = diff.clone().into_shape_with_order((1, diff.len())).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 512: `let diff_col = diff.clone().into_shape_with_order((diff.len(), 1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 513: `let diff_row = diff.clone().into_shape_with_order((1, diff.len())).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 526: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 530: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 728: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 732: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 737: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 741: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 764: `let std_dev = config.measurement_noise_scale.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 767: `noise[k] = rng.sample(rand_distr::Normal::new(0.0, std_dev).unwrap());`
  - **Fix**: Use .get() with proper bounds checking
- Line 885: `let output = (&denoised + &column_denoised) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 982: `output = (&output + &column_output) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1147: `let window_mean = window.mean().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1151: `/ window_size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1157: `let measurement_var = local_variances[local_variances.len() / 2];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1165: `let diff_mean = diff.iter().sum::<f64>() / diff.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1169: `/ diff.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1172: `let adjusted_process_var = process_var.min(measurement_var) / 10.0;`
  - **Fix**: Division without zero check - use safe_divide()

### src/lombscargle.rs

63 issues found:

- Line 264: `dts.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 266: `(dts[dts.len() / 2 - 1] + dts[dts.len() / 2]) / 2.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 268: `dts[dts.len() / 2]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 272: `let f_max = 0.5 * nyquist_factor / dt;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 276: `let f_min = 1.0 / t_range;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 284: `let n_samples = (t_range / dt).floor() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 290: `let mut freqs = Vec::with_capacity(n_freq / 2 + 1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 291: `for i in 0..=n_freq / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 292: `let f = f_min + (f_max - f_min) * (i as f64 / (n_freq / 2) as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 300: `let n_freq = (20.0 * t_range / dt).floor() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 305: `let f = f_min + (f_max - f_min) * (i as f64 / (n_freq - 1) as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 313: `let n_freq = (100.0 * (f_max / f_min).ln()).floor() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 313: `let n_freq = (100.0 * (f_max / f_min).ln()).floor() as usize;`
  - **Fix**: Mathematical operation .ln() without validation
- Line 319: `(f_min.ln()) + ((f_max / f_min).ln()) * (i as f64 / (n_freq - 1) as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 319: `(f_min.ln()) + ((f_max / f_min).ln()) * (i as f64 / (n_freq - 1) as f64);`
  - **Fix**: Mathematical operation .ln() without validation
- Line 342: `let mean = y.sum() / n_samples as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 374: `let tau = 0.5 * (s2omega / c2omega).atan2(1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 405: `let n1 = c_tau * c_tau / c_tau2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 406: `let n2 = s_tau * s_tau / s_tau2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 409: `(n1 + n2) / (yy - y_dot_h * y_dot_h / h_dot_h)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 412: `(c_tau * c_tau / c_tau2 + s_tau * s_tau / s_tau2) / y2_sum`
  - **Fix**: Division without zero check - use safe_divide()
- Line 425: `let denominator = yy - y_dot_h * y_dot_h / h_dot_h;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 428: `let n1 = c_tau * c_tau / c_tau2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 429: `let n2 = s_tau * s_tau / s_tau2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 432: `1.0 - (denominator - (n1 + n2)) / denominator`
  - **Fix**: Division without zero check - use safe_divide()
- Line 434: `1.0 - (y2_sum - (c_tau * c_tau / c_tau2 + s_tau * s_tau / s_tau2)) / y2_sum`
  - **Fix**: Division without zero check - use safe_divide()
- Line 447: `let n1 = c_tau * c_tau / c_tau2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 448: `let n2 = s_tau * s_tau / s_tau2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 451: `(n1 + n2) / (yy - y_dot_h * y_dot_h / h_dot_h)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 453: `(c_tau * c_tau / c_tau2 + s_tau * s_tau / s_tau2) / y2_sum`
  - **Fix**: Division without zero check - use safe_divide()
- Line 456: `standard.ln()`
  - **Fix**: Mathematical operation .ln() without validation
- Line 468: `let n1 = c_tau * c_tau / c_tau2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 469: `let n2 = s_tau * s_tau / s_tau2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 472: `0.5 * n_samples as f64 * (n1 + n2) / (yy - y_dot_h * y_dot_h / h_dot_h)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 474: `0.5 * n_samples as f64 * (c_tau * c_tau / c_tau2 + s_tau * s_tau / s_tau2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 475: `/ y2_sum`
  - **Fix**: Division without zero check - use safe_divide()
- Line 528: `.map(|&p| -(n_samples as f64) * (1.0 - p).ln())`
  - **Fix**: Mathematical operation .ln() without validation
- Line 531: `"psd" => power.iter().map(|&p| p * 2.0 / n_samples as f64).collect(),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 546: `let threshold = -fap.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 551: `"model" => 1.0 - (-threshold / n_samples as f64).exp(),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 552: `"log" => threshold.ln(),`
  - **Fix**: Mathematical operation .ln() without validation
- Line 553: `"psd" => threshold * n_samples as f64 / 2.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 618: `peak_indices.sort_by(|&a, &b| power[b].partial_cmp(&power[a]).unwrap());`
  - **Fix**: Use .get() with proper bounds checking
- Line 693: `.map(|i| freq_min + (freq_max - freq_min) * (i as f64) / (n_freqs - 1) as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 707: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 713: `.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 715: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 731: `let freqs_fft = autofrequency(&t, 1.0, Some(AutoFreqMethod::Fft)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 735: `let freqs_linear = autofrequency(&t, 1.0, Some(AutoFreqMethod::Linear)).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 739: `let freqs_log = autofrequency(&t, 1.0, Some(AutoFreqMethod::Log)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 744: `let nyquist = 0.5 / dt;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 748: `assert!(count_below_nyquist_fft > freqs_fft.len() * 9 / 10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 752: `assert!(count_below_nyquist_linear > freqs_linear.len() * 9 / 10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 755: `assert!(count_below_nyquist_log > freqs_log.len() * 9 / 10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 771: `let sig_standard = significance_levels(&power, &fap_levels, "standard", n_sample...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 772: `let sig_model = significance_levels(&power, &fap_levels, "model", n_samples).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 773: `let sig_log = significance_levels(&power, &fap_levels, "log", n_samples).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 774: `let sig_psd = significance_levels(&power, &fap_levels, "psd", n_samples).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 798: `let (peak_freqs, peak_powers) = find_peaks(&freq, &power, 0.5, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 809: `let (grouped_freqs, grouped_powers) = find_peaks(&freq, &power, 0.5, Some(0.15))...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 846: `.map(|i| freq_min + (freq_max - freq_min) * (i as f64) / (n_freqs - 1) as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 860: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 882: `peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()

### src/lombscargle_advanced_validation.rs

28 issues found:

- Line 273: `let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 276: `let f_nyquist = fs / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 285: `(freqs[nyquist_idx] - f_nyquist).abs() / f_nyquist`
  - **Fix**: Division without zero check - use safe_divide()
- Line 296: `let low_freq_precision = (freqs_low[low_idx] - f_low).abs() / f_low;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 413: `let true_freq = 0.1 / (2.0 * PI);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 414: `let accuracy = 1.0 - (detected_freq - true_freq).abs() / true_freq;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 419: `memory_samples.push(memory_usage as f64 / size as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 429: `performance_samples.last().unwrap() / performance_samples.first().unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 435: `1.0 / (memory_samples.iter().sum::<f64>() / memory_samples.len() as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 441: `accuracy_samples.iter().sum::<f64>() / accuracy_samples.len() as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 463: `(a - target).abs().partial_cmp(&(b - target).abs()).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 472: `.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 479: `let phase_true = PI / 4.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 502: `let amplitude_recovered = (2.0 * psd[peak_idx]).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 503: `let accuracy = 1.0 - (amplitude_recovered - amplitude_true).abs() / amplitude_tr...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 519: `let accuracy = 1.0 - (detected_freq - freq).abs() / freq;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 523: `Ok(accuracies.iter().sum::<f64>() / accuracies.len() as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 564: `sum_mag += (psd1[i] + psd2[i]) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 568: `1.0 - sum_diff / sum_mag`
  - **Fix**: Division without zero check - use safe_divide()
- Line 581: `let f_max = f1.last().unwrap().min(f5.last().unwrap()).min(f10.last().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 588: `let f = f_min + (f_max - f_min) * i as f64 / (n_samples - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 596: `let mean_p = (p1_interp + p5_interp + p10_interp) / 3.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 602: `1.0 - max_diff / mean_p`
  - **Fix**: Division without zero check - use safe_divide()
- Line 609: `Ok(agreements.iter().sum::<f64>() / agreements.len() as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 616: `let alpha = (f - freqs[i-1]) / (freqs[i] - freqs[i-1]);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 639: `let consistency = matched_peaks as f64 / peaks_no_window.len().max(1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 662: `let invariance = 1.0 - (detected_shift - shift).abs() / shift;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 730: `let base_score = (tests_passed as f64 / total_tests as f64) * 100.0;`
  - **Fix**: Division without zero check - use safe_divide()

### src/lombscargle_enhanced.rs

31 issues found:

- Line 179: `w[i] = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 186: `w[i] = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 193: `let x = 2.0 * PI * i as f64 / (n - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 219: `windowed[i] = values[i] * window[i] / window_sum.sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 219: `windowed[i] = values[i] * window[i] / window_sum.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 231: `let avg_dt = t_span / (n - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 232: `let nyquist = 0.5 / avg_dt;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 235: `let f_min = config.f_min.unwrap_or(1.0 / t_span);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 250: `frequencies.push(f_min + i as f64 * (f_max - f_min) / (n_freq - 1) as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 267: `let mean_val: f64 = values.iter().sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 271: `let t_mean = times.iter().sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 287: `let tau = 0.5 * sum_sin.atan2(sum_cos) / omega;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 314: `let variance: f64 = values_centered.iter().map(|&v| v * v).sum::<f64>() / n as f...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 318: `power[i] = 0.5 * ((c_tau * c_tau / c_tau2) + (s_tau * s_tau / s_tau2)) / varianc...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 337: `let mean_val: f64 = values.iter().sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 339: `let variance: f64 = values_centered.iter().map(|&v| v * v).sum::<f64>() / n as f...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 363: `power[i] = 0.5 * ((a * a / c) + (b * b / d)) / variance;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 400: `indices.sort_by(|&i, &j| boot_times[i].partial_cmp(&boot_times[j]).unwrap());`
  - **Fix**: Use .get() with proper bounds checking
- Line 417: `let lower_percentile = ((1.0 - confidence) / 2.0 * n_iterations as f64) as usize...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 418: `let upper_percentile = ((1.0 + confidence) / 2.0 * n_iterations as f64) as usize...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 425: `freq_powers.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 467: `1.0 - (1.0 - prob_single).powf(n_eff)`
  - **Fix**: Mathematical operation .powf( without validation
- Line 476: `let prob_single = (-chi2 / 2.0).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 477: `1.0 - (1.0 - prob_single).powf(n_frequencies as f64)`
  - **Fix**: Mathematical operation .powf( without validation
- Line 517: `let p_single = 1.0 - (1.0 - fap).powf(1.0 / n_frequencies as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 517: `let p_single = 1.0 - (1.0 - fap).powf(1.0 / n_frequencies as f64);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 518: `-(p_single.ln())`
  - **Fix**: Mathematical operation .ln() without validation
- Line 522: `let p_single = 1.0 - (1.0 - fap).powf(1.0 / n_frequencies as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 522: `let p_single = 1.0 - (1.0 - fap).powf(1.0 / n_frequencies as f64);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 523: `let chi2 = -2.0 * p_single.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 524: `chi2 / (n_samples - 3) as f64`
  - **Fix**: Division without zero check - use safe_divide()

### src/lombscargle_enhanced_validation.rs

26 issues found:

- Line 264: `let mean_time_ms = times.iter().sum::<f64>() / iterations as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 267: `.sum::<f64>() / iterations as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 268: `let std_time_ms = variance.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 318: `.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 319: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 323: `let freq_error = (peak_freq - f_true).abs() / f_true;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 327: `let avg_spacing = (t_irregular.last().unwrap() - t_irregular[0]) / (t_irregular....`
  - **Fix**: Use .get() with proper bounds checking
- Line 328: `let resolution_factor = 1.0 / avg_spacing;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 333: `let leakage_factor = 1.0 - (peak_power / total_power);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 383: `.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 384: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 388: `let estimated_amplitude = (2.0 * peak_power).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 391: `let frequency_error = (peak_freq - f_true).abs() / f_true;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 392: `let amplitude_error = (estimated_amplitude - a_true).abs() / a_true;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 426: `let noise_power = signal_power / 10.0_f64.powf(snr_db / 10.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 426: `let noise_power = signal_power / 10.0_f64.powf(snr_db / 10.0);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 427: `let noise_std = noise_power.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 456: `let threshold = power.iter().sum::<f64>() / power.len() as f64 * 3.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 465: `let detection_prob = detections as f64 / n_trials as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 527: `let normalized_power: Vec<f64> = power.iter().map(|&p| p / max_power).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 537: `(f1 - ref_freq).abs().partial_cmp(&(f2 - ref_freq).abs()).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 539: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 551: `let mean_absolute_error = deviations.iter().sum::<f64>() / deviations.len() as f...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 588: `distance / reference_peaks.len() as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 650: `let result = run_enhanced_validation("standard", &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 661: `let result = run_enhanced_validation("standard", &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/lombscargle_simd.rs

26 issues found:

- Line 211: `sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 267: `let windowed_view = ArrayView1::from_shape(n, &mut windowed).unwrap();`
  - **Fix**: Handle array creation errors properly
- Line 273: `windowed.iter_mut().for_each(|v| *v /= window_sum.sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 288: `dts.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 292: `(dts[(n - 1) / 2 - 1] + dts[(n - 1) / 2]) / 2.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 294: `dts[(n - 1) / 2]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 297: `let nyquist = 0.5 / median_dt;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 300: `let f_min = config.f_min.unwrap_or(1.0 / t_span);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 315: `frequencies.push(f_min + i as f64 * (f_max - f_min) / (n_freq - 1) as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 332: `let mean_val: f64 = values.iter().sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 335: `let centered_view = ArrayView1::from_shape(n, &mut values_centered).unwrap();`
  - **Fix**: Handle array creation errors properly
- Line 341: `let t_mean = times.iter().sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 344: `let shifted_view = ArrayView1::from_shape(n, &mut times_shifted).unwrap();`
  - **Fix**: Handle array creation errors properly
- Line 406: `0.5 * sin_sum.atan2(cos_sum) / omega`
  - **Fix**: Division without zero check - use safe_divide()
- Line 442: `let power = 0.5 * (c_tau * c_tau / c_tau2 + s_tau * s_tau / s_tau2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 471: `.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 477: `let mean_power: f64 = power.iter().sum::<f64>() / n_freq as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 482: `/ power.iter().filter(|&&p| (p - mean_power).abs() < mean_power).count() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 484: `let snr = max_power / noise_est.max(1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 487: `let half_max = max_power / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 511: `let leakage = 1.0 - max_power / window_power;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 566: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 584: `let alpha = (1.0 - confidence) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 593: `freq_powers.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 613: `1.0 - (1.0 - fap_single).powf(n_eff)`
  - **Fix**: Mathematical operation .powf( without validation
- Line 630: `let result = simd_lombscargle(&times, &values, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/lombscargle_validation.rs

14 issues found:

- Line 52: `let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 89: `.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 90: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 94: `let freq_error = (peak_freq - f_signal).abs() / f_signal;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 146: `power_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 195: `let rel_error = (power[i] - power_dc[i]).abs() / power[i].max(1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 211: `errors.iter().sum::<f64>() / errors.len() as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 217: `let stability_score = 1.0 - (issues.len() as f64 / 10.0).min(1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 321: `let stability_score = stability_tests_passed as f64 / total_tests as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 391: `.map(|&t| NumCast::from(t).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 395: `.map(|&v| NumCast::from(v).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 440: `let rel_diff = (power1[i] - power2[j]).abs() / power1[i].max(1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 455: `let result = validate_analytical_cases("standard", 0.01).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 462: `let result = validate_numerical_stability("standard").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/lti/analysis.rs

26 issues found:

- Line 52: `let log_step = f64::powf(w_max / w_min, 1.0 / (n - 1) as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 77: `let phase_deg = val.arg() * 180.0 / std::f64::consts::PI;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 691: `tf2.num[0] / tf1.num[0]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 697: `tf2.den[0] / tf1.den[0]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 754: `let frobenius_norm = norm_sum.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 844: `let factor = working_matrix[row][col] / working_matrix[rank][col];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 884: `dot_product(&orthogonal_col, basis_vec) / dot_product(basis_vec, basis_vec);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 928: `if diff_norm.sqrt() < tolerance {`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 960: `if diff_norm.sqrt() < tolerance {`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1026: `vec.iter().map(|x| x * x).sum::<f64>().sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1038: `let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1042: `let (w, mag, phase) = bode(&tf, Some(&freqs)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1071: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1073: `let analysis = analyze_controllability(&ss).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1089: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1091: `let analysis = analyze_observability(&ss).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1107: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1109: `let analysis = analyze_control_observability(&ss).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1120: `let tf1 = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1121: `let tf2 = TransferFunction::new(vec![2.0], vec![2.0, 2.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1123: `assert!(systems_equivalent(&tf1, &tf2, 1e-6).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1126: `let tf3 = TransferFunction::new(vec![1.0], vec![1.0, 2.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1127: `assert!(!systems_equivalent(&tf1, &tf3, 1e-6).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1134: `assert_eq!(matrix_rank(&identity).unwrap(), 2);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1138: `assert_eq!(matrix_rank(&singular).unwrap(), 1);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1146: `let result = matrix_multiply(&a, &b).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/lti/design.rs

22 issues found:

- Line 607: `let coeff = remainder[0] / divisor_lead;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 692: `let tf_sys = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 703: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 708: `let ss_sys = ss(vec![-1.0], vec![1.0], vec![1.0], vec![0.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 716: `let g1 = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 717: `let g2 = tf(vec![2.0], vec![1.0, 2.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 719: `let series_sys = series(&g1, &g2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 734: `let g1 = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 735: `let g2 = tf(vec![1.0], vec![1.0, 2.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 737: `let parallel_sys = parallel(&g1, &g2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 750: `let g = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 752: `let feedback_sys = feedback(&g, None, 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 766: `let g = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 767: `let h = tf(vec![2.0], vec![1.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 769: `let feedback_sys = feedback(&g, Some(&h as &dyn LtiSystem), 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 783: `let g = tf(vec![10.0], vec![1.0, 1.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 785: `let sens = sensitivity(&g, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 800: `let g = tf(vec![10.0], vec![1.0, 1.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 802: `let comp_sens = complementary_sensitivity(&g, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 849: `let (quotient, remainder) = divide_polynomials(&dividend, &divisor).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 883: `let g_ct = tf(vec![1.0], vec![1.0, 1.0], Some(false)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 884: `let g_dt = tf(vec![1.0], vec![1.0, 1.0], Some(true)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/lti/mod.rs

38 issues found:

- Line 135: `let tf_sys = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 136: `let _zpk_sys = zpk(Vec::new(), vec![Complex64::new(-1.0, 0.0)], 1.0, None).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 137: `let ss_sys = ss(vec![-1.0], vec![1.0], vec![1.0], vec![0.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 141: `let (w, mag, phase) = bode(&tf_sys, Some(&freqs)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 146: `let ctrl_analysis = analyze_controllability(&ss_sys).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 150: `let tf2 = tf(vec![2.0], vec![1.0, 2.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 151: `let series_sys = series(&tf_sys, &tf2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 152: `let parallel_sys = parallel(&tf_sys, &tf2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 153: `let feedback_sys = feedback(&tf_sys, None, 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 164: `let g1 = system::tf(vec![1.0], vec![1.0, 1.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 165: `let g2 = system::tf(vec![2.0], vec![1.0, 2.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 167: `let series_connection = system::series(&g1, &g2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 168: `let parallel_connection = system::parallel(&g1, &g2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 177: `let tf_sys = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 180: `let zpk_sys = tf_sys.to_zpk().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 181: `let ss_sys = tf_sys.to_ss().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 198: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 201: `let ctrl_analysis = analyze_controllability(&ss_sys).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 202: `let obs_analysis = analyze_observability(&ss_sys).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 203: `let combined_analysis = analyze_control_observability(&ss_sys).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 220: `let (wc, wo) = compute_lyapunov_gramians(&ss_sys).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 225: `let kalman_decomp = complete_kalman_decomposition(&ss_sys).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 271: `let g = tf(vec![10.0], vec![1.0, 1.0], None).unwrap(); // 10/(s+1)`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 273: `let s_func = sensitivity(&g, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 274: `let t_func = complementary_sensitivity(&g, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 288: `let sys1 = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 289: `let sys2 = tf(vec![2.0], vec![2.0, 2.0], None).unwrap(); // Same after normaliza...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 291: `assert!(systems_equivalent(&sys1, &sys2, 1e-6).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 294: `let sys3 = tf(vec![1.0], vec![1.0, 2.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 295: `assert!(!systems_equivalent(&sys1, &sys3, 1e-6).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 301: `let sys = tf(vec![1.0], vec![1.0, 1.0], None).unwrap(); // 1/(s+1)`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 304: `let response = sys.frequency_response(&freqs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 313: `assert_relative_eq!(response[1].norm(), 1.0 / (2.0_f64.sqrt()), epsilon = 1e-6);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 313: `assert_relative_eq!(response[1].norm(), 1.0 / (2.0_f64.sqrt()), epsilon = 1e-6);`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 319: `let _sys = tf(vec![1.0], vec![1.0, 1.0], None).unwrap(); // 1/(s+1)`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 324: `let sys_dt = tf(vec![1.0], vec![1.0, 1.0], Some(true)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 325: `let impulse = sys_dt.impulse_response(&t).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 326: `let step = sys_dt.step_response(&t).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/lti/systems.rs

28 issues found:

- Line 201: `num_val / den_val`
  - **Fix**: Division without zero check - use safe_divide()
- Line 294: `*x_i += ss.b[i * ss.n_inputs + j] * (1.0 / dt);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 357: `self.num[0] / self.den[0]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 403: `step[0] = impulse[0] * dt / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 407: `step[i] = step[i - 1] + (impulse[i - 1] + impulse[i]) * dt / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 545: `num / den`
  - **Fix**: Division without zero check - use safe_divide()
- Line 730: `let n_states = (a.len() as f64).sqrt() as usize;`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 740: `let n_inputs = if n_states == 0 { 0 } else { b.len() / n_states };`
  - **Fix**: Division without zero check - use safe_divide()
- Line 750: `let n_outputs = if n_states == 0 { 0 } else { c.len() / n_states };`
  - **Fix**: Division without zero check - use safe_divide()
- Line 933: `let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 945: `let tf = TransferFunction::new(vec![2.0], vec![2.0, 2.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 954: `let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 968: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 985: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 986: `assert!(zpk_stable.is_stable().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 990: `ZerosPoleGain::new(Vec::new(), vec![Complex64::new(1.0, 0.0)], 1.0, None).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 991: `assert!(!zpk_unstable.is_stable().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 996: `let ss = StateSpace::new(vec![-1.0], vec![1.0], vec![1.0], vec![0.0], None).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1013: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1015: `assert_eq!(ss.a(0, 0).unwrap(), -1.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1016: `assert_eq!(ss.a(0, 1).unwrap(), 0.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1017: `assert_eq!(ss.a(1, 0).unwrap(), 1.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1018: `assert_eq!(ss.a(1, 1).unwrap(), -2.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1020: `assert_eq!(ss.b(0, 0).unwrap(), 1.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1021: `assert_eq!(ss.b(1, 0).unwrap(), 0.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1023: `assert_eq!(ss.c(0, 0).unwrap(), 1.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1024: `assert_eq!(ss.c(0, 1).unwrap(), 0.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1026: `assert_eq!(ss.d(0, 0).unwrap(), 0.0);`
  - **Fix**: Replace with ? operator or .ok_or()

### src/lti_analysis_enhanced.rs

20 issues found:

- Line 193: `gram_eigenvalues[0] / gram_eigenvalues[gram_eigenvalues.len() - 1]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 242: `gram_eigenvalues[0] / gram_eigenvalues[gram_eigenvalues.len() - 1]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 398: `let q_vec = q.as_slice().unwrap().to_vec();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 613: `let s = lambda.ln() / dt;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 613: `let s = lambda.ln() / dt;`
  - **Fix**: Mathematical operation .ln() without validation
- Line 634: `let omega_n = (sigma * sigma + omega * omega).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 636: `-sigma / omega_n`
  - **Fix**: Division without zero check - use safe_divide()
- Line 642: `-1.0 / sigma`
  - **Fix**: Division without zero check - use safe_divide()
- Line 710: `let poles_set: std::collections::HashSet<_> = ss.a.eig().unwrap().0`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 769: `let target_mag = last_mag / 2.0_f64.sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 769: `let target_mag = last_mag / 2.0_f64.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 806: `let hankel_singular_values = eigenvalues.mapv(|lambda| lambda.norm().sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 896: `check_finite(&ss.a.as_slice().unwrap(), "A matrix")?;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 897: `check_finite(&ss.b.as_slice().unwrap(), "B matrix")?;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 898: `check_finite(&ss.c.as_slice().unwrap(), "C matrix")?;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 899: `check_finite(&ss.d.as_slice().unwrap(), "D matrix")?;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 912: `let u = u.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 946: `let vt = vt.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 988: `let result = analyze_controllability(&ss).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1003: `let result = analyze_stability(&ss).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/lti_response.rs

6 issues found:

- Line 264: `let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 270: `let response = impulse_response(&tf, &t).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 291: `let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 297: `let response = step_response(&tf, &t).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 323: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 330: `let y = lsim(&tf, &u, &t).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/measurements.rs

31 issues found:

- Line 53: `let mean_square = sum_of_squares / x_f64.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 54: `let rms = mean_square.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 155: `let peak_to_rms = peak / rms_val;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 236: `signal_f64.iter().map(|&x| x * x).sum::<f64>() / signal_f64.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 237: `let noise_power: f64 = noise.iter().map(|&x| x * x).sum::<f64>() / noise.len() a...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 248: `let snr_db = 10.0 * (signal_power / noise_power).log10();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 304: `if f0 <= 0.0 || f0 >= fs / 2.0 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 329: `let fft_bin_size = fs / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 332: `let f0_bin = (f0 / fft_bin_size).round() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 337: `let phi = 2.0 * std::f64::consts::PI * (f0_bin as f64 * k as f64) / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 349: `fundamental_power = (real * real + imag * imag) / (n * n) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 355: `let h_bin = (h as f64 * f0 / fft_bin_size).round() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 356: `if h_bin >= n / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 364: `let phi = 2.0 * std::f64::consts::PI * (h_bin as f64 * k as f64) / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 376: `h_power = (real * real + imag * imag) / (n * n) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 391: `let thd = (harmonic_power_sum / fundamental_power).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 391: `let thd = (harmonic_power_sum / fundamental_power).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 406: `let rms_val = rms(&dc_signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 411: `.map(|i| (2.0 * PI * i as f64 / 100.0).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 413: `let rms_val = rms(&sine_wave).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 414: `assert_relative_eq!(rms_val, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 414: `assert_relative_eq!(rms_val, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-2);`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 421: `let pp_val = peak_to_peak(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 426: `.map(|i| (2.0 * PI * i as f64 / 100.0).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 428: `let pp_val = peak_to_peak(&sine_wave).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 436: `let cf_val = peak_to_rms(&dc_signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 441: `.map(|i| (2.0 * PI * i as f64 / 100.0).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 443: `let cf_val = peak_to_rms(&sine_wave).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 444: `assert_relative_eq!(cf_val, 2.0_f64.sqrt(), epsilon = 1e-2);`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 451: `.map(|i| (2.0 * PI * i as f64 / 100.0).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 465: `let snr_db = snr(&clean, &noisy).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/median.rs

14 issues found:

- Line 124: `let half_kernel = kernel_size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 215: `let median_idx = weighted_window.len() / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 235: `let max_half_kernel = config.max_kernel_size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 343: `let half_kernel = kernel_size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 394: `let median_idx = flat_window.len() / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 454: `let median_idx = weighted_window.len() / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 475: `let max_half_kernel = config.max_kernel_size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 517: `let median_idx = flat_window.len() / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 632: `let half_kernel = kernel_size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 742: `sum_squared.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 784: `let half_kernel = kernel_size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 850: `let half_kernel = kernel_size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 910: `let plus_median = plus_shape[plus_shape.len() / 2];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 911: `let cross_median = cross_shape[cross_shape.len() / 2];`
  - **Fix**: Division without zero check - use safe_divide()

### src/multirate.rs

63 issues found:

- Line 147: `let phase = 2.0 * PI * k as f64 * (n as f64 - (L - 1) as f64 / 2.0) / M as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 180: `let phase = PI * k as f64 * (2.0 * n as f64 + 1.0) / (2.0 * M as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 208: `let center = (L - 1) as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 209: `let phase = PI * k as f64 * (n as f64 - center) / M as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 238: `let phase = 2.0 * PI * k as f64 * (n as f64 + 0.5) / M as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 244: `synthesis_filters[[k, n]] = (2.0 / M as f64) * analysis_filters[[k, n]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 271: `let phase = PI * (k as f64 + 0.5) * (n as f64 - (L - 1) as f64 / 2.0) / M as f64...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 274: `synthesis_filters[[k, n]] = (2.0 / M as f64) * analysis_filters[[k, n]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 284: `let cutoff = PI / num_channels as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 287: `let center = (length - 1) as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 292: `cutoff / PI`
  - **Fix**: Division without zero check - use safe_divide()
- Line 294: `(cutoff * t).sin() / (PI * t)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 320: `let scaling_factor = 2.0 / num_channels as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 330: `let cutoff = PI / num_channels as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 333: `let center = (length - 1) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 338: `cutoff / PI`
  - **Fix**: Division without zero check - use safe_divide()
- Line 340: `(cutoff * t).sin() / (PI * t)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 344: `let window_val = 0.54 - 0.46 * (2.0 * PI * n as f64 / (length - 1) as f64).cos()...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 366: `let overlap = length / num_channels;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 370: `let t = n as f64 - (length - 1) as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 371: `let normalized_t = t / overlap as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 389: `let arg = 2.0 * n as f64 / (length - 1) as f64 - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 391: `let i0_arg = Self::modified_bessel_i0(beta * (1.0 - arg * arg).sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 392: `i0_arg / i0_beta`
  - **Fix**: Division without zero check - use safe_divide()
- Line 401: `let i0_arg = Self::modified_bessel_i0(alpha * (1.0 - t * t).sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 402: `i0_arg / i0_alpha`
  - **Fix**: Division without zero check - use safe_divide()
- Line 409: `let t = x / 3.75;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 418: `let t = 3.75 / x.abs();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 427: `/ x.abs().sqrt()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 427: `/ x.abs().sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 438: `let polyphase_length = L / M;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 683: `error = error.sqrt() / min_len as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 683: `error = error.sqrt() / min_len as f64;`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 772: `for i in 0..num_freqs / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 796: `let _passband_end = num_freqs / (2 * M);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 797: `let stopband_start = num_freqs / M;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 821: `let passband_end = num_freqs / (2 * M);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 914: `let nyquist_freq = PI / (upsampling_factor.max(downsampling_factor) as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 932: `let center = (length - 1) as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 937: `cutoff / PI`
  - **Fix**: Division without zero check - use safe_divide()
- Line 939: `(cutoff * t).sin() / (PI * t)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 943: `let window_val = 0.42 - 0.5 * (2.0 * PI * n as f64 / (length - 1) as f64).cos()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 944: `+ 0.08 * (4.0 * PI * n as f64 / (length - 1) as f64).cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1028: `self.upsampling_factor as f64 / self.downsampling_factor as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1040: `PerfectReconstructionFilterBank::new(config, PrFilterDesign::Orthogonal).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1057: `PerfectReconstructionFilterBank::new(config, PrFilterDesign::Orthogonal).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1062: `let subbands = filter_bank.analysis(&input).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1066: `let reconstructed = filter_bank.synthesis(&subbands).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1081: `PerfectReconstructionFilterBank::new(config, PrFilterDesign::Orthogonal).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1085: `let (error, _is_perfect) = filter_bank.verify_perfect_reconstruction(&impulse).u...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1102: `PerfectReconstructionFilterBank::new(config, PrFilterDesign::Biorthogonal).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1105: `let subbands = filter_bank.analysis(&input).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1106: `let reconstructed = filter_bank.synthesis(&subbands).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1122: `PerfectReconstructionFilterBank::new(config, PrFilterDesign::LinearPhase).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1135: `let mut converter = MultirateConverter::new(3, 2, 32).unwrap(); // Smaller filte...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1138: `let output = converter.convert(&input).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1157: `PerfectReconstructionFilterBank::new(config, PrFilterDesign::Orthogonal).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1159: `let properties = filter_bank.analyze_properties().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1182: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1185: `let subbands = filter_bank.analysis(&input).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1186: `let reconstructed = filter_bank.synthesis(&subbands).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1202: `PerfectReconstructionFilterBank::new(config, PrFilterDesign::ModulatedDft).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1209: `let subbands = filter_bank.analysis(&input).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/multitaper/adaptive.rs

10 issues found:

- Line 122: `nfft_val / 2 + 1`
  - **Fix**: Division without zero check - use safe_divide()
- Line 149: `f.push(i as f64 * fs_val / nfft_val as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 156: `if i <= nfft_val / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 157: `f.push(i as f64 * fs_val / nfft_val as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 159: `f.push((i as f64 - nfft_val as f64) * fs_val / nfft_val as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 183: `s_initial[j] = weighted_sum / weight_sum;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 197: `denominator += numerator.powi(2) / (s_initial[j] + 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 198: `weights[[i, j]] = numerator / (s_initial[j] + 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 228: `1.0 / fs_val`
  - **Fix**: Division without zero check - use safe_divide()
- Line 249: `1.0 / (fs_val * weight_sum)`
  - **Fix**: Division without zero check - use safe_divide()

### src/multitaper/dpss_enhanced.rs

17 issues found:

- Line 56: `let w = nw / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 67: `eigenvalues[j].abs().partial_cmp(&eigenvalues[i].abs()).unwrap()`
  - **Fix**: Use .get() with proper bounds checking
- Line 106: `let term = (n as f64 - 1.0 - 2.0 * i as f64) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 113: `.map(|i| (i as f64 * (n - i) as f64) / 2.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 151: `let norm = eigvec.dot(eigvec).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 169: `let mid = n / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 219: `.map(|c| c.re / fft_size as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 229: `(lag as f64 * 2.0 * PI * w).sin() / (lag as f64 * 2.0 * PI * w)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 248: `let ratios = ratios.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 289: `let norm = tapers.row(i).dot(&tapers.row(i)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 317: `let ratios = ratios.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 341: `let (tapers, ratios) = dpss_enhanced(64, 4.0, 7, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 350: `let (tapers, _) = dpss_enhanced(128, 4.0, 7, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 363: `let (tapers, _) = dpss_enhanced(128, 4.0, 7, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 374: `let (_, ratios) = dpss_enhanced(64, 4.0, 7, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 375: `let ratios = ratios.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 390: `assert!(validate_dpss_implementation().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()

### src/multitaper/enhanced.rs

32 issues found:

- Line 209: `let tapered_view = ArrayView1::from_shape(n, &mut tapered).unwrap();`
  - **Fix**: Handle array creation errors properly
- Line 250: `let spectrum = simd_fft(&tapered, nfft).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 302: `let n_freqs = nfft / 2 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 304: `.map(|i| i as f64 * fs / nfft as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 309: `if i <= nfft / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 310: `i as f64 * fs / nfft as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 312: `(i as f64 - nfft as f64) * fs / nfft as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 319: `let n_freqs = if onesided { nfft / 2 + 1 } else { nfft };`
  - **Fix**: Division without zero check - use safe_divide()
- Line 324: `2.0 / (fs * weight_sum)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 326: `1.0 / (fs * weight_sum)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 349: `let n_freqs = if onesided { nfft / 2 + 1 } else { nfft };`
  - **Fix**: Division without zero check - use safe_divide()
- Line 381: `psd[j] = weighted_sum / weight_sum;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 388: `let bias_factor = 1.0 / (1.0 + (psd[j] / spectra[[i, j]]).powi(2));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 396: `.map(|(old, new)| ((old - new) / old.max(1e-10)).abs())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 407: `.map(|i| i as f64 * fs / nfft as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 412: `if i <= nfft / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 413: `i as f64 * fs / nfft as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 415: `(i as f64 - nfft as f64) * fs / nfft as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 422: `let scaling = if onesided { 2.0 / fs } else { 1.0 / fs };`
  - **Fix**: Division without zero check - use safe_divide()
- Line 446: `let lower_quantile = chi2.inverse_cdf(alpha / 2.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 447: `let upper_quantile = chi2.inverse_cdf(1.0 - alpha / 2.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 449: `let lower_factor = dof / upper_quantile;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 450: `let upper_factor = dof / lower_quantile;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 464: `let psd_estimate = weighted_sum / weight_sum;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 478: `2.0 * sum_lambda.powi(2) / sum_lambda_sq`
  - **Fix**: Division without zero check - use safe_divide()
- Line 545: `let n_windows = (n - window_size) / step + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 558: `.map(|i| (i * step + window_size / 2) as f64 / config.fs)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 563: `&& n_windows >= config.multitaper.parallel_threshold / window_size {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 574: `enhanced_pmtm(window, &mt_config).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 632: `.map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / 100.0).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 636: `let result = enhanced_pmtm(&signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 645: `let result = simd_fft(&signal, 4).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/multitaper/ftest.rs

6 issues found:

- Line 97: `let f_stat = (numerator * dof2) / (denominator * dof1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 186: `let y_mean = sum_wy / sum_w;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 187: `let u_mean = sum_wu / sum_w;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 201: `beta = numerator / denominator;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 225: `let f_stat = (mss / dof1) / (rss / dof2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 334: `let avg_f = combined_f / valid_harmonics as f64;`
  - **Fix**: Division without zero check - use safe_divide()

### src/multitaper/jackknife.rs

26 issues found:

- Line 60: `psd_full[j] = weighted_sum / weight_sum;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 80: `let estimate = weighted_sum / weight_sum;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 82: `estimate.ln()`
  - **Fix**: Mathematical operation .ln() without validation
- Line 100: `let t_critical = t_dist.inverse_cdf(1.0 - (1.0 - conf_level) / 2.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 105: `psd_full[j].ln()`
  - **Fix**: Mathematical operation .ln() without validation
- Line 122: `jack_var *= (k - 1) as f64 / k as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 125: `let se = jack_var.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 209: `let renorm_weight = adaptive_weights[[i, j]] / weight_sum;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 215: `jackknife_estimates[[i_out, j]] = weighted_sum.ln(); // Log transform`
  - **Fix**: Mathematical operation .ln() without validation
- Line 227: `let t_critical = t_dist.inverse_cdf(1.0 - (1.0 - conf_level) / 2.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 230: `let full_log = psd_full[j].ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 244: `jack_var *= (k - 1) as f64 / k as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 246: `let se = jack_var.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 316: `coherence[j] = sxy.norm_sqr() / (sxx * syy);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 348: `jack_coh[[i_out, j]] = sxy.norm_sqr() / (sxx * syy);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 359: `let t_critical = t_dist.inverse_cdf(1.0 - (1.0 - conf_level) / 2.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 367: `z_vals[i] = 0.5 * ((1.0 + coh.sqrt()) / (1.0 - coh.sqrt())).ln();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 367: `z_vals[i] = 0.5 * ((1.0 + coh.sqrt()) / (1.0 - coh.sqrt())).ln();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 367: `z_vals[i] = 0.5 * ((1.0 + coh.sqrt()) / (1.0 - coh.sqrt())).ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 371: `let z_mean: f64 = z_vals.iter().sum::<f64>() / k as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 374: `.sum::<f64>() * (k - 1) as f64 / k as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 375: `let z_se = z_var.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 381: `coherence_lower[j] = ((2.0 * z_lower).exp() - 1.0) / ((2.0 * z_lower).exp() + 1....`
  - **Fix**: Division without zero check - use safe_divide()
- Line 382: `coherence_upper[j] = ((2.0 * z_upper).exp() - 1.0) / ((2.0 * z_upper).exp() + 1....`
  - **Fix**: Division without zero check - use safe_divide()
- Line 398: `phase_var *= (k - 1) as f64 / k as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 399: `let phase_se = phase_var.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation

### src/multitaper/psd.rs

21 issues found:

- Line 149: `let n_freqs = nfft_val / 2 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 152: `f.push(i as f64 * fs_val / nfft_val as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 159: `if i <= nfft_val / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 160: `f.push(i as f64 * fs_val / nfft_val as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 162: `f.push((i as f64 - nfft_val as f64) * fs_val / nfft_val as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 170: `nfft_val / 2 + 1`
  - **Fix**: Division without zero check - use safe_divide()
- Line 177: `let n_freqs = nfft_val / 2 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 206: `let scaling = 1.0 / (fs_val * weight_sum);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 309: `let step_val = step.unwrap_or(window_size_val / 2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 317: `let n_segments = (n - window_size_val) / step_val + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 319: `nfft_val / 2 + 1`
  - **Fix**: Division without zero check - use safe_divide()
- Line 328: `let center = (i * step_val + window_size_val / 2) as f64 / fs_val;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 337: `f.push(i as f64 * fs_val / nfft_val as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 344: `if i <= nfft_val / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 345: `f.push(i as f64 * fs_val / nfft_val as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 347: `f.push((i as f64 - nfft_val as f64) * fs_val / nfft_val as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 421: `s_initial[m] = weighted_sum / weight_sum;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 435: `denominator += numerator.powi(2) / (s_initial[m] + 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 436: `weights[[j, m]] = numerator / (s_initial[m] + 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 466: `1.0 / fs_val`
  - **Fix**: Division without zero check - use safe_divide()
- Line 487: `1.0 / (fs_val * weight_sum)`
  - **Fix**: Division without zero check - use safe_divide()

### src/multitaper/utils.rs

15 issues found:

- Line 158: `let mut x_spectra = Array2::zeros((k_val, nfft_val / 2 + 1));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 159: `let mut y_spectra = Array2::zeros((k_val, nfft_val / 2 + 1));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 160: `let mut cross_spectra = Array2::zeros((k_val, nfft_val / 2 + 1));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 177: `let n_freqs = nfft_val / 2 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 188: `let n_freqs = nfft_val / 2 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 191: `f.push(i as f64 * fs_val / nfft_val as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 202: `let n_freqs = nfft_val / 2 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 218: `let denominator = (x_power * y_power).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 220: `coherence[j] = (cross_power / denominator).powi(2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 409: `let mut even = Vec::with_capacity(n / 2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 410: `let mut odd = Vec::with_capacity(n / 2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 424: `for k in 0..n / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 427: `(-2.0 * PI * k as f64 / n as f64).cos(),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 428: `(-2.0 * PI * k as f64 / n as f64).sin(),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 432: `result[k + n / 2] = even_fft[k] - t;`
  - **Fix**: Division without zero check - use safe_divide()

### src/multitaper/validation.rs

23 issues found:

- Line 240: `let t: Vec<f64> = (0..test_signals.n).map(|i| i as f64 / test_signals.fs).collec...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 263: `let snr_linear = 10.0_f64.powf(test_signals.snr_db / 10.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 263: `let snr_linear = 10.0_f64.powf(test_signals.snr_db / 10.0);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 264: `let noise_std = 1.0 / snr_linear.sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 264: `let noise_std = 1.0 / snr_linear.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 274: `let peak_idx = (freq * test_signals.n as f64 / test_signals.fs) as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 279: `let mean_estimate = peak_values.iter().sum::<f64>() / peak_values.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 281: `let bias = (mean_estimate - true_power).abs() / true_power;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 285: `.sum::<f64>() / (peak_values.len() - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 410: `let simd_speedup = standard_time_ms / enhanced_serial_time;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 411: `let parallel_speedup = enhanced_serial_time / enhanced_time_ms;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 431: `let t: Vec<f64> = (0..test_signals.n).map(|i| i as f64 / test_signals.fs).collec...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 461: `let rel_error = (ref_val - enh_val).abs() / ref_val;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 467: `let mean_relative_error = relative_errors.iter().sum::<f64>() / relative_errors....`
  - **Fix**: Division without zero check - use safe_divide()
- Line 491: `let half_power = peak_power / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 516: `(total_power - lobe_power) / total_power`
  - **Fix**: Division without zero check - use safe_divide()
- Line 522: `max_val / min_val`
  - **Fix**: Division without zero check - use safe_divide()
- Line 532: `let mean_x = x.iter().sum::<f64>() / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 533: `let mean_y = y.iter().sum::<f64>() / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 547: `cov / (var_x * var_y).sqrt()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 547: `cov / (var_x * var_y).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 575: `Ok(coverage_count as f64 / n_trials as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 595: `score -= spectral.variance.sqrt() * 50.0;`
  - **Fix**: Mathematical operation .sqrt() without validation

### src/multitaper/windows.rs

10 issues found:

- Line 84: `((n_float - 1.0) / 2.0 - i_float).powi(2) * (2.0 * std::f64::consts::PI * w).pow...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 100: `idx.sort_by(|&i, &j| eigvals[i].partial_cmp(&eigvals[j]).unwrap());`
  - **Fix**: Use .get() with proper bounds checking
- Line 119: `lambda[i] = (1.0 - sorted_eigvals[i]).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 123: `let norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 242: `a[i - 1] = a[i - 1] + t * (r * r - (a[i - 1] - g).powi(2) / 4.0 - f * f) / r;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 243: `a[i] = g + t * (r * r - (a[i - 1] - g).powi(2) / 4.0 - f * f) / r;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 245: `c = r / ((a[i - 1] - g).powi(2) / 4.0 + f * f).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 245: `c = r / ((a[i - 1] - g).powi(2) / 4.0 + f * f).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 246: `s = f / ((a[i - 1] - g).powi(2) / 4.0 + f * f).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 246: `s = f / ((a[i - 1] - g).powi(2) / 4.0 + f * f).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation

### src/nlm.rs

53 issues found:

- Line 122: `let half_patch = config.patch_size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 123: `let half_search = config.search_window / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 144: `let sigma_d = half_search as f64 / 3.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 148: `kernel[i] = (-d / (2.0 * sigma_d.powi(2))).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 216: `let weight = (-dist / h_adjusted).exp() * dist_weight;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 238: `denoised[i] = weighted_sum / weight_sum;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 286: `let half_patch = config.patch_size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 287: `let half_search = config.search_window / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 314: `let sigma_g = config.patch_size as f64 / 6.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 321: `patch_weights[[i, j]] = (-0.5 * (di + dj) / sigma_g.powi(2)).exp() / gauss_norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 391: `let spatial_dist = (di + dj).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 392: `let spatial_sigma = half_search as f64 / 3.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 393: `(-spatial_dist / (2.0 * spatial_sigma.powi(2))).exp()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 399: `let weight = (-dist / h_adjusted).exp() * dist_weight;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 428: `denoised[[i, j]] = weighted_sum / weight_sum;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 472: `let half_patch = config.patch_size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 473: `let half_search = config.search_window / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 527: `let spatial_dist = (di + dj).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 528: `let spatial_sigma = half_search as f64 / 3.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 529: `(-spatial_dist / (2.0 * spatial_sigma.powi(2))).exp()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 535: `let weight = (-dist / h_adjusted).exp() * dist_weight;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 561: `block_weights.iter().map(|&w| w / total_weight).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 639: `scale_config.patch_size = cmp::max(3, config.patch_size / scale_factor);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 642: `config.search_window / scale_factor,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 653: `let scaled_height = height / scale_factor;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 654: `let scaled_width = width / scale_factor;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 669: `let scale_weight = 1.0 / (2.0_f64.powi(scale as i32));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 681: `.map(|s| 1.0 / (2.0_f64.powi(s as i32)))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 683: `result.mapv_inplace(|x| x / total_weight);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 709: `let half_patch = config.patch_size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 710: `let half_search = config.search_window / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 730: `let avg_sigma = channel_sigma.iter().sum::<f64>() / channels as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 811: `let spatial_dist = (di + dj).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 812: `let spatial_sigma = half_search as f64 / 3.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 813: `(-spatial_dist / (2.0 * spatial_sigma.powi(2))).exp()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 819: `let weight = (-total_dist / h_adjusted).exp() * dist_weight;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 845: `denoised[[i, j, c]] = weighted_sums[c] / weight_sum;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 950: `let half_size = size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 977: `let half_size = size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1028: `sum_diff_sq / n as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1062: `sum_diff_sq / sum_weights`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1076: `sum_diff_sq / n as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1100: `(diffs[diffs.len() / 2 - 1] + diffs[diffs.len() / 2]) / 2.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1102: `diffs[diffs.len() / 2]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1107: `median / 0.6745 / std::f64::consts::SQRT_2`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1137: `(laplacian[laplacian.len() / 2 - 1] + laplacian[laplacian.len() / 2]) / 2.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1139: `laplacian[laplacian.len() / 2]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1143: `median / 0.6745 / std::f64::consts::SQRT_2`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1164: `let h_scale = height as f64 / new_height as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1165: `let w_scale = width as f64 / new_width as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1188: `downsampled[[i, j]] = sum / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1214: `let h_scale = (height - 1) as f64 / (new_height - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1215: `let w_scale = (width - 1) as f64 / (new_width - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()

### src/parallel_spectral.rs

62 issues found:

- Line 172: `let n_freq_bins = window_size / 2 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 176: `.map(|i| i as f64 * fs / window_size as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 217: `csd / n_time_frames as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 236: `csd / n_time_frames as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 381: `let mut planner = self.fft_planner.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 388: `let n_freq_bins = nfft_actual / 2 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 394: `psd[i] = if i == 0 || (i == nfft_actual / 2 && nfft_actual % 2 == 0) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 395: `magnitude_sq / normalization`
  - **Fix**: Division without zero check - use safe_divide()
- Line 397: `2.0 * magnitude_sq / normalization`
  - **Fix**: Division without zero check - use safe_divide()
- Line 403: `.map(|i| i as f64 * fs / nfft_actual as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 427: `.map(|i| i as f64 * fs / (2 * (n_freq_bins - 1)) as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 431: `.map(|i| i as f64 * hop_size as f64 / fs)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 451: `let n_frames = (n_samples - window_size) / hop_size + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 452: `let n_freq_bins = window_size / 2 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 464: `let chunk_size = (n_frames / num_threads()).max(1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 594: `(pxy.norm_sqr()) / (pxx * pyy)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 618: `(pxy.norm_sqr()) / (pxx * pyy)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 627: `.map(|i| i as f64 * fs / (2 * (n_freq_bins - 1)) as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 642: `let beta = 2.0 * std::f64::consts::PI * nw * taper_idx as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 645: `let t = (i as f64 - n as f64 / 2.0) / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 646: `let w = 2.0 * nw / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 652: `(w * std::f64::consts::PI * t).sin() / (std::f64::consts::PI * t)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 653: `} * (1.0 - 2.0 * taper_idx as f64 / k as f64).max(0.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 657: `let norm: f64 = taper.iter().map(|&x| x * x).sum::<f64>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 677: `let n_freq_bins = n / 2 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 718: `avg_psd[i] += value / tapers.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 783: `let n_segments = (n - window_size) / hop_size + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 798: `let n_freq_bins = window_size / 2 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 828: `avg_psd[i] += if i == 0 || (i == window_size / 2 && window_size % 2 == 0) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 829: `magnitude_sq / normalization`
  - **Fix**: Division without zero check - use safe_divide()
- Line 831: `2.0 * magnitude_sq / normalization`
  - **Fix**: Division without zero check - use safe_divide()
- Line 844: `.map(|i| i as f64 * fs / window_size as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1108: `let mut cross_psd_sum = vec![Complex64::new(0.0, 0.0); window_size / 2 + 1];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1148: `.map(|i| i as f64 * fs / window_size as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1167: `phase_band.1 / (fs / 2.0),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1172: `amplitude_band.1 / (fs / 2.0),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1200: `let mean_vector_length = (sum_real * sum_real + sum_imag * sum_imag).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1201: `/ (amplitudes.iter().sum::<f64>() / n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1221: `let n_frames = (n - window_size) / hop_size + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1222: `let n_freqs = window_size / 2 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1264: `cross_spectrum.norm_sqr() / (auto1 * auto2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1276: `.map(|i| i as f64 * fs / window_size as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1280: `.map(|i| i as f64 * hop_size as f64 / fs)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1302: `let probabilities: Vec<f64> = psd.iter().map(|&p| p / total_power).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1309: `.map(|&p| p * p.ln())`
  - **Fix**: Mathematical operation .ln() without validation
- Line 1317: `.map(|&p| p * p.ln())`
  - **Fix**: Mathematical operation .ln() without validation
- Line 1323: `.map(|&p| p.powf(q))`
  - **Fix**: Mathematical operation .powf( without validation
- Line 1325: `(1.0 / (1.0 - q)) * sum_q.ln()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1325: `(1.0 / (1.0 - q)) * sum_q.ln()`
  - **Fix**: Mathematical operation .ln() without validation
- Line 1332: `.map(|&p| p.powf(q))`
  - **Fix**: Mathematical operation .powf( without validation
- Line 1334: `(1.0 - sum_q) / (q - 1.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1391: `.map(|i| (2.0 * PI * 50.0 * i as f64 / fs).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1394: `.map(|i| (2.0 * PI * 100.0 * i as f64 / fs).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1402: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1418: `let t = i as f64 / fs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1429: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1447: `.map(|i| (2.0 * PI * 50.0 * i as f64 / fs).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1460: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1470: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1482: `.map(|i| (2.0 * PI * 50.0 * i as f64 / fs).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1485: `.map(|i| (2.0 * PI * 100.0 * i as f64 / fs).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1490: `let results = parallel_welch(&signals, fs, 512, 0.5, Some("hann")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/parametric.rs

23 issues found:

- Line 138: `autocorr[lag] = sum / (n - lag) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 190: `let k_reflection = (autocorr[k + 1] - err) / e;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 253: `let mut e = signal.iter().map(|&x| x * x).sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 271: `let k_m = -2.0 * num / den;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 580: `let norm_freqs = freqs.mapv(|f| f * 2.0 * PI / fs);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 596: `psd[i] = variance / h.norm_sqr();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 657: `r[k] = sum / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 674: `ma_coeffs[k] = (r[k] - sum) / v[0];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 736: `let norm_freqs = freqs.mapv(|f| f * 2.0 * PI / fs);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 759: `psd[i] = variance * b.norm_sqr() / a.norm_sqr();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 801: `if max_order >= signal.len() / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 815: `let variance = signal.iter().map(|&x| x * x).sum::<f64>() / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 819: `OrderSelection::AIC => criteria[order] = n * variance.ln() + 2.0,`
  - **Fix**: Mathematical operation .ln() without validation
- Line 820: `OrderSelection::BIC => criteria[order] = n * variance.ln() + (0 as f64).ln() * n...`
  - **Fix**: Mathematical operation .ln() without validation
- Line 821: `OrderSelection::FPE => criteria[order] = variance * (n + 1.0) / (n - 1.0),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 823: `criteria[order] = n * variance.ln() + 0.5 * (0 as f64).ln() * n`
  - **Fix**: Mathematical operation .ln() without validation
- Line 825: `OrderSelection::AICc => criteria[order] = n * variance.ln() + 2.0,`
  - **Fix**: Mathematical operation .ln() without validation
- Line 835: `criteria[order] = n * variance.ln() + 2.0 * order as f64;`
  - **Fix**: Mathematical operation .ln() without validation
- Line 838: `criteria[order] = n * variance.ln() + order as f64 * n.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 841: `criteria[order] = variance * (n + order as f64) / (n - order as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 844: `criteria[order] = n * variance.ln() + 0.5 * order as f64 * n.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 849: `n * variance.ln() + 2.0 * order as f64 * (n / (n - order as f64 - 1.0));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 849: `n * variance.ln() + 2.0 * order as f64 * (n / (n - order as f64 - 1.0));`
  - **Fix**: Mathematical operation .ln() without validation

### src/parametric_adaptive.rs

11 issues found:

- Line 208: `let k = px / denominator;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 216: `model.gain = (&model.gain - &outer_product.dot(&model.gain)) / lambda;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 245: `let k = px / s;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 280: `let normalized_step = step_size / (norm_sq + epsilon);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 363: `let f = i as f64 / (2.0 * n_freq as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 376: `let power = model.variance / h.norm_sqr();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 425: `peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 446: `let model = initialize_adaptive_ar(&config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 460: `let mut model = initialize_adaptive_ar(&config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 466: `let error = update_adaptive_ar(&mut model, sample, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 485: `let (freqs, psd) = adaptive_spectrum(&model, 128).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/parametric_arma.rs

19 issues found:

- Line 89: `let ar_order = ((n as f64).sqrt() as usize).max(p + q);`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 148: `let variance = residuals.mapv(|r| r * r).sum() / (n - p - q) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 245: `let variance = innovations.mapv(|e| e * e).sum() / (n - p - q) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 262: `let variance = residuals.mapv(|r| r * r).sum() / (n - p - q) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 311: `ar_coeffs[i] -= step_size * grad / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 322: `ma_coeffs[i] -= step_size * grad / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 330: `let variance = residuals.mapv(|r| r * r).sum() / (n - p - q) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 351: `let log_likelihood = -0.5 * n as f64 * (2.0 * PI * model.variance).ln()`
  - **Fix**: Mathematical operation .ln() without validation
- Line 352: `- 0.5 * residuals.mapv(|r| r * r).sum() / model.variance;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 413: `r[k] = sum / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 489: `let factor = aug[[k, i]] / aug[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 529: `let w = 2.0 * PI * freq / fs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 546: `psd[i] = model.variance * ma_poly.norm_sqr() / ar_poly.norm_sqr();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 608: `extended_signal = extended_signal.clone().into_shape(n + h + 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 611: `extended_residuals = extended_residuals.clone().into_shape(n + h + 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 615: `forecast_errors[h] = model.variance.sqrt() * (1.0 + h as f64 / 10.0).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 615: `forecast_errors[h] = model.variance.sqrt() * (1.0 + h as f64 / 10.0).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 627: `let z = inverse_normal_cdf((1.0 + conf) / 2.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 663: `x * num / den`
  - **Fix**: Division without zero check - use safe_divide()

### src/parametric_enhanced.rs

31 issues found:

- Line 165: `let max_ar = config.max_ar_order.min((n / 4).max(1));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 166: `let max_ma = config.max_ma_order.min((n / 4).max(1));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 221: `if p + q < n / 3 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 239: `.min_by(|a, b| a.bic.partial_cmp(&b.bic).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 270: `if p + q < signal.len() / 3 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 331: `let bic = -2.0 * log_likelihood + (k as f64) * (n as f64).ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 332: `let fpe = variance * (n as f64 + k as f64) / (n as f64 - k as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 333: `let mdl = -log_likelihood + 0.5 * (k as f64) * (n as f64).ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 334: `let aicc = aic + 2.0 * k as f64 * (k as f64 + 1.0) / (n as f64 - k as f64 - 1.0)...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 365: `let log_likelihood = -0.5 * n as f64 * (2.0 * PI * variance).ln()`
  - **Fix**: Mathematical operation .ln() without validation
- Line 366: `- 0.5 * sum_sq_residuals / variance;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 391: `let signal_segment = &signal.as_slice().unwrap()[t-p..t];`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 392: `let ar_segment = &ar.as_slice().unwrap()[1..=p];`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 408: `residuals.copy_from_slice(signal.as_slice().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 441: `let avg_aic: f64 = all_results.iter().map(|r| r.aic).sum::<f64>() / all_results....`
  - **Fix**: Division without zero check - use safe_divide()
- Line 442: `let avg_bic: f64 = all_results.iter().map(|r| r.bic).sum::<f64>() / all_results....`
  - **Fix**: Division without zero check - use safe_divide()
- Line 469: `.map(|i| i as f64 / (2.0 * (n_freq - 1) as f64))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 531: `psd[i] = variance / magnitude_sq;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 574: `psd[i] = variance * ma_magnitude_sq / ar_magnitude_sq;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 596: `let lower_quantile = chi2.inverse_cdf(alpha / 2.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 597: `let upper_quantile = chi2.inverse_cdf(1.0 - alpha / 2.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 599: `let lower_factor = dof / upper_quantile;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 600: `let upper_factor = dof / lower_quantile;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 617: `let residual_variance = residuals.iter().map(|&r| r * r).sum::<f64>() / residual...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 647: `lb_stat += (acf[k] * acf[k]) / (n - k) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 667: `let mean = signal.iter().sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 672: `let centered_view = ArrayView1::from_shape(n, &mut centered).unwrap();`
  - **Fix**: Handle array creation errors properly
- Line 677: `let variance = f64::simd_dot(&centered_view, &centered_view) / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 691: `acf[lag] = sum / (n as f64 * variance);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 720: `let result = enhanced_parametric_estimation(&signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 738: `let result = enhanced_parametric_estimation(&signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/parametric_validation.rs

42 issues found:

- Line 255: `order_selection_accuracy: order_accuracy_sum / total_tests as f64,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 256: `coefficient_error: coefficient_errors.iter().sum::<f64>() / coefficient_errors.l...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 257: `prediction_error: prediction_errors.iter().sum::<f64>() / prediction_errors.len(...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 258: `spectral_accuracy: 1.0 - spectral_errors.iter().sum::<f64>() / spectral_errors.l...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 259: `model_stable: stability_count as f64 / total_tests as f64 > 0.95,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 302: `let var_error = (estimated.variance - true_var).abs() / true_var;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 308: `likelihood_improvements.push((ll - null_ll) / signal.len() as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 323: `ar_coefficient_accuracy: 1.0 - ar_errors.iter().sum::<f64>() / ar_errors.len() a...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 324: `ma_coefficient_accuracy: 1.0 - ma_errors.iter().sum::<f64>() / ma_errors.len() a...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 325: `variance_accuracy: 1.0 - variance_errors.iter().sum::<f64>() / variance_errors.l...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 327: `/ likelihood_improvements.len().max(1) as f64,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 328: `identifiable: identifiable_count as f64 / total_tests as f64 > 0.9,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 357: `arma_agreements.push((ar_agreement + ma_agreement) / 2.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 375: `yw_burg_agreement: yw_burg_agreements.iter().sum::<f64>() / yw_burg_agreements.l...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 376: `arma_method_agreement: arma_agreements.iter().sum::<f64>() / arma_agreements.len...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 378: `/ spectral_consistencies.len() as f64,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 380: `/ parameter_consistencies.len() as f64,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 422: `let sensitivity = compute_coefficient_error(&ar_clean, &ar_noisy) * (10.0_f64.po...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 422: `let sensitivity = compute_coefficient_error(&ar_clean, &ar_noisy) * (10.0_f64.po...`
  - **Fix**: Mathematical operation .powf( without validation
- Line 442: `noise_sensitivity: noise_sensitivities.iter().sum::<f64>() / noise_sensitivities...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 444: `/ outlier_robustness_scores.len() as f64,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 475: `(t2 / t1) / (n2 / n1)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 487: `estimation_time_ms: times.iter().sum::<f64>() / times.len() as f64,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 567: `innovations[i] = rng.random_range(-1.0..1.0) * variance.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 592: `let signal_power = signal.iter().map(|&x| x * x).sum::<f64>() / signal.len() as ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 593: `let noise_power = signal_power / 10.0_f64.powf(snr_db / 10.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 593: `let noise_power = signal_power / 10.0_f64.powf(snr_db / 10.0);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 594: `let noise_std = noise_power.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 614: `(error / n as f64).sqrt()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 614: `(error / n as f64).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 630: `(error / (n - order) as f64).sqrt() / variance.sqrt()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 630: `(error / (n - order) as f64).sqrt() / variance.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 639: `error += ((psd_true[i] - psd_est[i]) / psd_true[i]).powi(2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 642: `Ok((error / psd_true.len() as f64).sqrt())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 642: `Ok((error / psd_true.len() as f64).sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 650: `let mean = (psd1[i] + psd2[i]) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 655: `1.0 - (sum_sq_diff / sum_sq_mean).sqrt()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 655: `1.0 - (sum_sq_diff / sum_sq_mean).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 679: `let aic = n * variance.ln() + 2.0 * order as f64;`
  - **Fix**: Mathematical operation .ln() without validation
- Line 692: `let variance = signal.iter().map(|&x| x * x).sum::<f64>() / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 693: `-0.5 * n * (2.0 * PI * variance).ln() - 0.5 * n`
  - **Fix**: Mathematical operation .ln() without validation
- Line 726: `let ratio = r[0] / r[order].abs().max(1e-10);`
  - **Fix**: Division without zero check - use safe_divide()

### src/peak.rs

9 issues found:

- Line 409: `left_ip = x1 + (x2 - x1) * (height - y1) / (y2 - y1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 425: `right_ip = x1 + (x2 - x1) * (height - y1) / (y2 - y1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 452: `let peaks = find_peaks(&signal, None, None, None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 456: `let peaks = find_peaks(&signal, Some(1.5), None, None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 466: `let peaks = find_peaks(&signal, None, Some(0.8), None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 486: `let peaks = find_peaks(&signal, None, None, Some(2), None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 509: `let peaks = find_peaks(&signal, None, None, None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 515: `let prominences = peak_prominences(&signal, &peaks).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 533: `let (widths, _left_ips, _right_ips) = peak_widths(&signal, &peaks, Some(0.5)).un...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/performance_optimized.rs

21 issues found:

- Line 85: `let n_full_chunks = out_size / simd_width;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 97: `"same" => (k - 1) / 2,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 124: `"same" => (k - 1) / 2,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 167: `let n_chunks = (n + chunk_size - overlap - 1) / (chunk_size - overlap);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 231: `let b_norm: Array1<f64> = b / a0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 232: `let a_norm: Array1<f64> = a / a0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 242: `let n_chunks = (n + chunk_size - overlap - 1) / (chunk_size - overlap);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 347: `let b_norm = b / a0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 348: `let a_norm = a / a0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 422: `let tile_size = (config.cache_line_size * 4) / std::mem::size_of::<f64>();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 426: `(0..((out_rows + tile_size - 1) / tile_size))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 432: `for tile_col in 0..((out_cols + tile_size - 1) / tile_size) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 452: `for row_tile in 0..((out_rows + tile_size - 1) / tile_size) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 456: `for col_tile in 0..((out_cols + tile_size - 1) / tile_size) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 531: `let approx_size = (n + 1) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 532: `let detail_size = (n + 1) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 656: `(-((i as f64 - kernel_size as f64 / 2.0).powi(2)) / 10.0).exp()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 673: `1.0 - max_error / result_standard.iter().map(|&x| x.abs()).fold(0.0, f64::max);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 680: `speedup: standard_time / optimized_time,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 696: `let result = simd_convolve_1d(&signal, &kernel, "same").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 705: `let mut filter = StreamingFilter::new(b, a, 1024).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/phase_vocoder.rs

24 issues found:

- Line 121: `let synthesis_hop = (analysis_hop as f64 / config.time_stretch).round() as usize...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 158: `.map(|k| 2.0 * PI * k as f64 / config.window_size as f64 * analysis_hop as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 186: `let true_freq = bin_frequencies[k] + deviation / analysis_hop as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 248: `((num_frames as f64) * (synthesis_hop as f64) / (analysis_hop as f64)) as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 298: `let pitch_factor = 2.0_f64.powf(pitch_shift_semitones / 12.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 298: `let pitch_factor = 2.0_f64.powf(pitch_shift_semitones / 12.0);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 312: `time_stretch: 1.0 / pitch_factor,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 365: `let pos = i as f64 / factor;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 452: `formant_envelope[k] / formant_envelope[warped_bin]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 458: `new_frame[k] = Complex64::from_polar(magnitude * correction_factor.sqrt(), phase...`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 534: `let idx = i.saturating_sub(smoothing_width / 2).saturating_add(j);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 542: `*smooth = sum / count;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 618: `*w = 0.5 * (1.0 - (2.0 * PI * n as f64 / (length - 1) as f64).cos());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 623: `*w = 0.54 - 0.46 * (2.0 * PI * n as f64 / (length - 1) as f64).cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 628: `*w = 0.42 - 0.5 * (2.0 * PI * n as f64 / (length - 1) as f64).cos()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 629: `+ 0.08 * (4.0 * PI * n as f64 / (length - 1) as f64).cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 676: `let scale = 1.0 / signal.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 696: `.map(|i| (2.0 * PI * freq * i as f64 / sample_rate as f64).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 708: `let result = phase_vocoder(&signal, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 722: `let hann = create_window("hann", n).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 728: `for i in 0..n / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 737: `assert_relative_eq!(hann[n / 2], 1.0, epsilon = 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 746: `let upsampled = resample(&signal, factor).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 767: `let downsampled = resample(&signal, factor).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/realtime.rs

34 issues found:

- Line 88: `self.data.len() / self.channels`
  - **Fix**: Division without zero check - use safe_divide()
- Line 150: `let mut buffer = self.buffer.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 157: `*self.overrun_count.lock().unwrap() += 1;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 168: `let mut buffer = self.buffer.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 186: `self.buffer.lock().unwrap().len()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 191: `*self.overrun_count.lock().unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 196: `self.buffer.lock().unwrap().clear();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 233: `let mut running = self.running.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 262: `*self.running.lock().unwrap() = false;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 286: `self.stats.lock().unwrap().clone()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 291: `*self.stats.lock().unwrap() = RealtimeStats::default();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 305: `while *running.lock().unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 310: `let mut input_buf = input_buffer.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 327: `let mut stats_guard = stats.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 347: `let mut output_buf = output_buffer.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 359: `let mut stats_guard = stats.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 377: `let samples_per_ms = config.sample_rate / 1000.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 378: `let buffer_latency = (input_buffer.lock().unwrap().len()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 379: `+ output_buffer.lock().unwrap().len())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 381: `/ samples_per_ms;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 388: `let target_block_time = config.buffer_size as f64 / config.sample_rate * 1000.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 552: `*sample = self.sum / self.history.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 558: `self.window_size / 2`
  - **Fix**: Division without zero check - use safe_divide()
- Line 598: `self.lookahead_samples = (self.attack_time * config.sample_rate / 1000.0) as usi...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 614: `let attack_coeff = (-1.0 / (self.attack_time * self.sample_rate / 1000.0)).exp()...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 615: `let release_coeff = (-1.0 / (self.release_time * self.sample_rate / 1000.0)).exp...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 624: `self.threshold / sample_abs`
  - **Fix**: Division without zero check - use safe_divide()
- Line 680: `buffer.write(&input_data).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 683: `let read_count = buffer.read(&mut output_data).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 695: `buffer.write(&input_data).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 721: `processor.process_block(&mut block).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 731: `processor.process_block(&mut block).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 748: `let ch0 = block.channel_data(0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 749: `let ch1 = block.channel_data(1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/reassigned.rs

23 issues found:

- Line 57: `hop_size: window_size / 4,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 145: `let center = (win.len() - 1) as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 159: `fw[i] = (win[i + 1] - win[i - 1]) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 186: `let hop_seconds = config.hop_size as f64 / config.fs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 190: `let frequencies = Array1::linspace(0.0, config.fs / 2.0, n_freqs);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 217: `signal.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 259: `let omega = 2.0 * PI * i as f64 / (2.0 * (n_bins - 1) as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 267: `let inst_phase_derivative = (stft_time[[i, j]] / stft_val).im;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 271: `let group_delay = (stft_freq[[i, j]] / stft_val).im;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 284: `freq_shifts[[i, j]] = freq_shifts[[i, j]] * fs / (2.0 * PI);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 314: `let f_reassigned = freq_shifts[[i, j]] / ((n_bins - 1) as f64) * (n_bins as f64)...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 377: `let half_width = width / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 410: `smoothed[[i, j]] = sum / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 488: `let avg_energy_a = a.iter().map(|(t, _)| *t).sum::<usize>() as f64 / a.len() as ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 489: `let avg_energy_b = b.iter().map(|(t, _)| *t).sum::<usize>() as f64 / b.len() as ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 518: `let duration = n as f64 / fs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 525: `window: Array1::from(window::hann(128, true).unwrap()),`
  - **Fix**: Handle array creation errors properly
- Line 533: `let result = reassigned_spectrogram(&signal, config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 568: `window: Array1::from(window::hann(64, true).unwrap()),`
  - **Fix**: Handle array creation errors properly
- Line 575: `let standard = reassigned_spectrogram(&signal, config.clone()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 576: `let smoothed = smoothed_reassigned_spectrogram(&signal, config, 3).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 601: `let freq_bin_45hz = (45.0 / (fs / 2.0) * standard.frequencies.len() as f64) as u...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 602: `let freq_bin_55hz = (55.0 / (fs / 2.0) * standard.frequencies.len() as f64) as u...`
  - **Fix**: Division without zero check - use safe_divide()

### src/resample.rs

22 issues found:

- Line 91: `let up = up / gcd;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 92: `let down = down / gcd;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 95: `let n_out = ((x_f64.len() * up) as f64 / down as f64).ceil() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 105: `let t = i as f64 - (filter_length - 1) as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 109: `(std::f64::consts::PI * t).sin() / (std::f64::consts::PI * t)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 116: `* (2.0 * std::f64::consts::PI * i as f64 / (filter_length - 1) as f64).cos()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 120: `- (2.0 * std::f64::consts::PI * i as f64 / (filter_length - 1) as f64).cos())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 126: `let w = 2.0 * std::f64::consts::PI * i as f64 / (filter_length - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 133: `*h_val = sinc * window_val / up as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 138: `let output_time = i as f64 * down as f64 / up as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 146: `for j in 0..filter_length / up {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 323: `let (up, down) = rational_approximation(num as f64 / x.len() as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 371: `let error = ((num as f64 / denom as f64) - x).abs();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 382: `(best_num / d, best_denom / d)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 394: `let resampled = resample(&signal, 1, 1, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 406: `let upsampled = upsample(&signal, 2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 423: `let downsampled = downsample(&signal, 2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 426: `assert!(downsampled.len() >= signal.len() / 2 - 1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 427: `assert!(downsampled.len() <= signal.len() / 2 + 1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 436: `let resampled = resample_poly(&signal, 150).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 440: `let resampled = resample_poly(&signal, 50).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 448: `let approx = num as f64 / denom as f64;`
  - **Fix**: Division without zero check - use safe_divide()

### src/robust.rs

30 issues found:

- Line 118: `let half_window = window_size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 150: `window_values.push(*window_values.last().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 155: `window_values.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 159: `let trimmed_mean = trimmed_values.iter().sum::<f64>() / trimmed_values.len() as ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 214: `let half_window = window_size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 235: `window_values.push(*window_values.last().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 241: `sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 243: `let mid = sorted_values.len() / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 244: `(sorted_values[mid - 1] + sorted_values[mid]) / 2.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 246: `sorted_values[sorted_values.len() / 2]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 252: `abs_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 255: `let mid = abs_deviations.len() / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 256: `(abs_deviations[mid - 1] + abs_deviations[mid]) / 2.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 258: `abs_deviations[abs_deviations.len() / 2]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 322: `let half_window = window_size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 342: `window_values.push(*window_values.last().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 346: `window_values.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 349: `let lower_idx = ((percentile / 100.0) * (window_values.len() - 1) as f64) as usi...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 351: `(((100.0 - percentile) / 100.0) * (window_values.len() - 1) as f64) as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 417: `let half_window = window_size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 437: `window_values.push(*window_values.last().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 465: `weighted_sum / weight_sum`
  - **Fix**: Division without zero check - use safe_divide()
- Line 538: `let filtered = alpha_trimmed_filter(&signal, 3, 0.3).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 550: `let (filtered, outliers) = hampel_filter(&signal, 3, 3.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 560: `let filtered = winsorize_filter(&signal, 5, 20.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 570: `let filtered = huber_filter(&signal, 3, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 583: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 585: `let filtered = robust_filter_2d(&image, alpha_trimmed_filter, 3, 0.2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 596: `let result = alpha_trimmed_filter(&empty_signal, 3, 0.2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 601: `let result = alpha_trimmed_filter(&small_signal, 3, 0.2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/savgol.rs

32 issues found:

- Line 98: `let halflen = window_length / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 123: `wl * (wl * wl - 1.0) / 12.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 135: `/ (4.0 * norm_factor);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 142: `coeffs[i] = -x / (2.0 * norm_factor / 3.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 148: `coeffs[i] = 1.0 / (norm_factor / 3.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 217: `y[deriv_val] = fact / delta_val.powi(deriv_val as i32);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 279: `let mid = nrows / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 287: `let norm = (nrows_f64 * (nrows_f64 * nrows_f64 - 1.0)) / 12.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 289: `(3.0 * nrows_f64 * nrows_f64 - 7.0 - 30.0 * x * x) / (4.0 * norm);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 294: `let norm = (nrows_f64 * (nrows_f64 * nrows_f64 - 1.0)) / 12.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 295: `output[i] = (-x) / (2.0 * norm / 3.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 300: `let norm = (nrows_f64 * (nrows_f64 * nrows_f64 - 1.0)) / 12.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 301: `output[i] = 1.0 / (norm / 3.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 362: `let factor = aug[[j, i]] / aug[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 377: `x[i] = (aug[[i, n]] - sum) / aug[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 501: `values.push(result / config.delta.powi(config.deriv as i32));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 531: `let halflen = window_length / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 687: `let halflen = window_length / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 719: `let halflen = window_length / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 831: `let coeffs = savgol_coeffs(5, 2, None, None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 843: `let coeffs = savgol_coeffs(5, 2, Some(1), None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 855: `let t: Vec<f64> = (0..100).map(|i| i as f64 / 10.0).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 860: `*val += 0.1 * (i as f64 / 5.0).sin();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 864: `let smoothed = savgol_filter(&x, 11, 2, None, None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 875: `let half_win = window_size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 893: `let mean = x.iter().sum::<f64>() / x.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 894: `let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 904: `let interp = savgol_filter(&x, 5, 2, None, None, Some("interp"), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 905: `let mirror = savgol_filter(&x, 5, 2, None, None, Some("mirror"), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 906: `let constant = savgol_filter(&x, 5, 2, None, None, Some("constant"), None).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 907: `let nearest = savgol_filter(&x, 5, 2, None, None, Some("nearest"), None).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 908: `let wrap = savgol_filter(&x, 5, 2, None, None, Some("wrap"), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/separation.rs

6 issues found:

- Line 227: `.map(|&x| x * config.separation_power.sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 232: `.map(|&x| x * (2.0 - config.separation_power).sqrt().max(0.1))`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 251: `.map(|i| i as f64 / sample_rate)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 269: `let bands = multiband_separation(&signal_array, &cutoffs, sample_rate, None).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 300: `.map(|i| i as f64 / sample_rate)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 317: `harmonic_percussive_separation(&signal_array, sample_rate, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/simd_ops.rs

7 issues found:

- Line 21: `SIMD_CAPS.as_ref().unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 242: `(sum_squares / signal.len() as f32).sqrt()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 242: `(sum_squares / signal.len() as f32).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 365: `envelope[i] = s.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 381: `let start = (kernel_len - 1) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 418: `simd_energy_f32(signal) / signal.len() as f32`
  - **Fix**: Division without zero check - use safe_divide()
- Line 535: `let start = (kernel_len - 1) / 2;`
  - **Fix**: Division without zero check - use safe_divide()

### src/sparse.rs

34 issues found:

- Line 106: `min(m / 4, n.saturating_sub(1))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 236: `min(m / 3, n.saturating_sub(1))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 253: `col.mapv_inplace(|val| val / norm);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 347: `let step_size = 1.0 / (phi_norm * phi_norm);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 386: `let diff = x_diff_norm / x_norm.max(config.eps);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 419: `let step_size = 1.0 / (phi_norm * phi_norm);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 457: `t_next = (1.0 + f64::sqrt(1.0 + 4.0 * t * t)) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 460: `z = &x + ((t - 1.0) / t_next) * (&x - &x_prev);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 470: `let diff = x_diff_norm / x_norm.max(config.eps);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 502: `min(m / 4, n.saturating_sub(1))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 532: `proxy_values.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 588: `temp_values.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 607: `let diff = x_diff_norm / x_norm.max(config.eps);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 658: `min(m / 3, n.saturating_sub(1))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 702: `values.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 718: `let diff = x_diff_norm / x_norm.max(config.eps);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 760: `min(m / 3, n.saturating_sub(1))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 782: `initial_values.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 822: `corr_values.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 865: `merged_values.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 973: `x_current.mapv(|val| -val * f64::exp(-0.5 * (val / sigma).powi(2)) / sigma.powi(...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1218: `result[i] = (complex_signal[i].re.powi(2) + complex_signal[i].im.powi(2)).sqrt()...`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1239: `let scale = 1.0 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1289: `for i in (0..n_rows).step_by(patch_size / 2) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1290: `for j in (0..n_cols).step_by(patch_size / 2) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1351: `let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1364: `let norm = col.mapv(|x| x * x).sum().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1366: `col.mapv_inplace(|x| x / norm);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1398: `let coherence = (inner_product / (norm_i * norm_j)).abs();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1454: `x.mapv_inplace(|val| val / x_norm);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1505: `Ok(1.0 - (l1_norm / l2_norm_val) / (n as f64).sqrt())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1505: `Ok(1.0 - (l1_norm / l2_norm_val) / (n as f64).sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1553: `let scale = 1.0 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1618: `let threshold = y.fold(0.0, |acc, &val| acc + val.abs()) / (n as f64) * config.l...`
  - **Fix**: Division without zero check - use safe_divide()

### src/spectral.rs

35 issues found:

- Line 35: `* (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (nperseg - 1) as f64).cos());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 44: `- 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (nperseg - 1) as f64).cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 53: `- 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (nperseg - 1) as f64).cos()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 54: `+ 0.08 * (4.0 * std::f64::consts::PI * i as f64 / (nperseg - 1) as f64).cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 81: `let mean = x.iter().sum::<f64>() / x.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 99: `let slope = (n as f64 * sum_xy - sum_x * sum_y) / (n as f64 * sum_xx - sum_x * s...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 100: `let intercept = (sum_y - slope * sum_x) / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 192: `let scale = 1.0 / win_scale;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 207: `.map(|&c| c.norm_sqr() * scale / (fs_val * x_f64.len() as f64))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 211: `let freqs = scirs2_fft::helper::fftfreq(nfft_val, 1.0 / fs_val).map_err(|e| {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 288: `let noverlap_val = noverlap.unwrap_or(nperseg_val / 2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 319: `let scale = 1.0 / win_scale;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 324: `(x_f64.len() - noverlap_val) / step`
  - **Fix**: Division without zero check - use safe_divide()
- Line 336: `let freqs = scirs2_fft::helper::fftfreq(nfft_val, 1.0 / fs_val).map_err(|e| {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 383: `.map(|&c| c.norm_sqr() * scale / (fs_val * nperseg_val as f64))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 425: `let pad_len = nperseg / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 433: `let pad_len = nperseg / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 500: `let noverlap_val = noverlap.unwrap_or(nperseg_val / 2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 542: `(input_signal.len() - noverlap_val) / step`
  - **Fix**: Division without zero check - use safe_divide()
- Line 554: `let freqs = scirs2_fft::helper::fftfreq(nfft_val, 1.0 / fs_val).map_err(|e| {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 565: `let center = i * step + nperseg_val / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 566: `times.push(center as f64 / fs_val);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 686: `let scale = 1.0 / win_scale;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 693: `let psd = c.norm_sqr() * scale / (fs_val * nperseg_val as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 747: `let t: Vec<f64> = (0..1000).map(|i| i as f64 / fs).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 752: `let (freqs, psd) = periodogram(&x, Some(fs), None, None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 768: `assert!(freqs.len() >= (x.len() / 2));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 776: `let t: Vec<f64> = (0..2000).map(|i| i as f64 / fs).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 788: `welch(&x, Some(fs), None, Some(256), Some(128), None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 808: `let t: Vec<f64> = (0..2000).map(|i| i as f64 / fs).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 826: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 869: `let t: Vec<f64> = (0..1000).map(|i| i as f64 / fs).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 884: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 896: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 908: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/spline.rs

62 issues found:

- Line 148: `result[i] = (xi.powi(3)) / 6.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 150: `result[i] = (-3.0 * xi.powi(3) + 12.0 * xi.powi(2) - 12.0 * xi + 4.0) / 6.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 152: `result[i] = (3.0 * xi.powi(3) - 24.0 * xi.powi(2) + 60.0 * xi - 44.0) / 6.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 154: `result[i] = (4.0 - xi).powi(3) / 6.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 162: `result[i] = xi.powi(4) / 24.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 167: `/ 24.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 172: `/ 24.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 177: `/ 24.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 179: `result[i] = (5.0 - xi).powi(4) / 24.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 187: `result[i] = xi.powi(5) / 120.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 193: `/ 120.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 199: `/ 120.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 205: `/ 120.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 211: `/ 120.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 213: `result[i] = (6.0 - xi).powi(5) / 120.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 250: `let num = vec![1.0 / 8.0, 3.0 / 4.0, 1.0 / 8.0];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 256: `let num = vec![1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 257: `let den = vec![1.0, -2.0 / 3.0];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 263: `1.0 / 384.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 264: `19.0 / 96.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 265: `115.0 / 192.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 266: `19.0 / 96.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 267: `1.0 / 384.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 275: `1.0 / 120.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 276: `13.0 / 60.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 277: `11.0 / 20.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 278: `13.0 / 60.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 279: `1.0 / 120.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 325: `sum / (1.0 - z_n)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 336: `sum / (1.0 - z_i)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 710: `let c0 = (1.0 - t).powi(3) / 6.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 711: `let c1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 712: `let c2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 713: `let c3 = t3 / 6.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 721: `let c1 = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 722: `let c2 = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 723: `let c3 = t3 / 6.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 728: `let c0 = (1.0 - t).powi(3) / 6.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 729: `let c1 = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 730: `let c2 = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 750: `.and_then(|sum| sum.checked_sub(order_int / 2))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 753: `let basis_x = (t + (j as f64) - (order_int as f64) / 2.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 754: `+ order_int as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 834: `let mean = signal_f64.iter().sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 843: `let gamma = 1.0 / (1.0 + lam * h.powi(2 * order.as_int() as i32));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 985: `let basis = bspline_basis(&x, SplineOrder::Cubic).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 997: `let filtered = bspline_filter(&signal, SplineOrder::Cubic).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1009: `let filtered_ramp = bspline_filter(&ramp, SplineOrder::Cubic).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1022: `let coeffs = bspline_coefficients(&signal, SplineOrder::Cubic).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1039: `let coeffs = bspline_coefficients(&signal, SplineOrder::Cubic).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1043: `let values = bspline_evaluate(&coeffs, &x, SplineOrder::Cubic).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1052: `let values_mid = bspline_evaluate(&coeffs, &x_mid, SplineOrder::Cubic).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1060: `let values_half = bspline_evaluate(&coeffs, &x_half, SplineOrder::Cubic).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1074: `let smoothed = bspline_smooth(&signal, SplineOrder::Cubic, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1083: `let no_smooth = bspline_smooth(&signal, SplineOrder::Cubic, 0.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1089: `let high_smooth = bspline_smooth(&signal, SplineOrder::Cubic, 1e7).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1090: `let mean = signal.iter().sum::<f64>() / signal.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1102: `let coeffs = bspline_coefficients(&signal, SplineOrder::Cubic).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1107: `let deriv = bspline_derivative(&coeffs, &x1, SplineOrder::Cubic, 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1117: `let deriv2 = bspline_derivative(&coeffs, &x2, SplineOrder::Cubic, 2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1128: `let mean = x.iter().sum::<f64>() / x.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1129: `let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()

### src/sswt.rs

13 issues found:

- Line 196: `let phase_diff = (next.arg() - prev.arg()) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 208: `center_frequency / scale / 2.0 / PI + phase_diff_unwrapped / 2.0 / PI`
  - **Fix**: Division without zero check - use safe_divide()
- Line 210: `center_frequency / scale`
  - **Fix**: Division without zero check - use safe_divide()
- Line 226: `omega[[i, 0]] = center_frequency / scale / 2.0 / PI + phase_diff / 2.0 / PI;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 228: `omega[[i, 0]] = center_frequency / scale;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 242: `center_frequency / scale / 2.0 / PI + phase_diff / 2.0 / PI;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 244: `omega[[i, n_samples - 1]] = center_frequency / scale;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 292: `sst[[freq_idx, t]] += cwt_val / scale.sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 292: `sst[[freq_idx, t]] += cwt_val / scale.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 343: `let min_log = min_scale.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 344: `let max_log = max_scale.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 563: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 571: `assert_eq!(result.cwt.unwrap().shape()[0], 32); // Scales`
  - **Fix**: Replace with ? operator or .ok_or()

### src/stft.rs

97 issues found:

- Line 257: `let m_num_mid = m_num / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 365: `*w_i /= dd_i.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 446: `calc_dual_window_internal(self.win.as_slice().unwrap(), self.hop)`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 455: `1.0 / self.fs`
  - **Fix**: Division without zero check - use safe_divide()
- Line 473: `1.0 / (self.mfft as f64 * self.t())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 482: `-((self.m_num - self.m_num_mid) as isize / self.hop as isize)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 495: `((n + self.m_num_mid) as isize - 1) / self.hop as isize + 1`
  - **Fix**: Division without zero check - use safe_divide()
- Line 549: `self.mfft / 2 + 1`
  - **Fix**: Division without zero check - use safe_divide()
- Line 574: `if i <= self.mfft / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 585: `let half = self.mfft / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 646: `let half = self.mfft / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 762: `ScalingMode::Magnitude => 1.0 / self.win.map(|x| x * x).sum().sqrt(),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 762: `ScalingMode::Magnitude => 1.0 / self.win.map(|x| x * x).sum().sqrt(),`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 763: `ScalingMode::Psd => 1.0 / self.win.map(|x| x * x).sum(),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 904: `for i in 1..(self.mfft / 2) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 909: `for i in 1..(self.mfft / 2 + 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 917: `Complex64::new(2.0_f64.sqrt(), 0.0)`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 923: `let nyquist_idx = if self.mfft % 2 == 0 { self.mfft / 2 } else { 0 };`
  - **Fix**: Division without zero check - use safe_divide()
- Line 945: `let half = self.mfft / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 984: `for k in 1..(n / 2 + 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 987: `let angle = -2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1004: `let angle = -2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1021: `let angle = -2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1029: `let half = n / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1046: `let phase_factor = phase_shift as f64 / self.mfft as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1084: `for k in 1..(n / 2 + 1).min(spectrum.len()) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1085: `let angle = 2.0 * std::f64::consts::PI * (i * k) as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1108: `let angle = 2.0 * std::f64::consts::PI * (i * k) as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1127: `let half = n / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1144: `let angle = 2.0 * std::f64::consts::PI * (i * k) as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1189: `((self.m_num - self.m_num_mid) as isize + self.hop as isize - 1) / self.hop as i...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1190: `self.m_num_mid as isize / self.hop as isize + 1,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1194: `(n as isize - self.m_num_mid as isize) / self.hop as isize,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1196: `/ (self.hop as isize)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1246: `let dual_win = Array1::from_vec(win.iter().zip(dd.iter()).map(|(&w, &d)| w / d)....`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1321: `let numerator = (q_d.iter().map(|&x| x * x).sum::<f64>()).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1330: `let alpha = numerator / denominator;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1366: `let window = window::hann(256, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1377: `let stft = ShortTimeFft::new(&window, 64, 1000.0, Some(config)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1399: `let stft = ShortTimeFft::from_window("hamming", 1000.0, 256, 192, Some(config))....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1415: `let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1421: `let hann_window = window::hann(window_length, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1431: `let stft = ShortTimeFft::new(&hann_window, hop_size, fs, Some(config)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1434: `let stft_result = stft.stft(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1440: `let freq_bin_100hz = (100.0 / stft.delta_f()).round() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1458: `let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1469: `let stft = ShortTimeFft::from_win_equals_dual(&window, hop_size, fs, Some(config...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1472: `let stft_result = stft.stft(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1475: `let reconstructed = stft.istft(&stft_result, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1497: `let avg_error = error_sum / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1512: `let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1523: `let hann_window = window::hann(window_length, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1533: `let stft = ShortTimeFft::new(&hann_window, hop_size, fs, Some(config)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1536: `let spec_result = stft.spectrogram(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1542: `let freq_bin_100hz = (100.0 / stft.delta_f()).round() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1543: `let freq_bin_250hz = (250.0 / stft.delta_f()).round() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1623: `1_000_000 / (fft_size * 8)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1626: `1_000_000 / (fft_size * 16)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1675: `overlap / hop_size`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1737: `overlap / hop_size`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1771: `let frames_in_chunk = chunk_size / self.stft.hop + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1778: `memory_per_chunk as f64 / 1_000_000.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1787: `let signal_memory_mb = std::mem::size_of_val(signal) / 1_000_000;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1789: `if signal_memory_mb > self.config.max_memory_mb / 4 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1803: `let signal_memory_mb = std::mem::size_of_val(signal) / 1_000_000;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1805: `if signal_memory_mb > self.config.max_memory_mb / 4 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1858: `let skip_frames = if i == 0 { 0 } else { overlap / hop_size };`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1889: `let frames_per_chunk = chunk_size / hop_size + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1893: `total_frames as f64 * self.stft.f_pts() as f64 * 8.0 / 1_000_000.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1895: `total_frames as f64 * self.stft.f_pts() as f64 * 16.0 / 1_000_000.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1899: `frames_per_chunk as f64 * self.stft.f_pts() as f64 * 8.0 / 1_000_000.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1901: `frames_per_chunk as f64 * self.stft.f_pts() as f64 * 16.0 / 1_000_000.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1912: `memory_reduction_factor: total_memory_mb / chunk_memory_mb,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1949: `let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1954: `let window = window::hann(window_length, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1966: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1969: `let result = mem_stft.stft_chunked(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1986: `let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1991: `let window = window::hann(window_length, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2003: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2005: `let spec_result = mem_stft.spectrogram_chunked(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2029: `let window = window::hann(window_length, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2041: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2044: `let small_result = mem_stft.stft_auto(&small_signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2045: `let large_result = mem_stft.stft_auto(&large_signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2063: `let window = window::hann(window_length, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2075: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2100: `let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2105: `let window = window::hann(window_length, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2117: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2120: `let parallel_result = mem_stft.stft_parallel_chunked(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2121: `let sequential_result = mem_stft.stft_chunked(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2146: `let t = i as f64 / fs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2154: `let window = window::hann(window_length, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2166: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2179: `let spec_result = mem_stft.spectrogram_auto(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2186: `let expected_freq_bins = window_length / 2 + 1;`
  - **Fix**: Division without zero check - use safe_divide()

### src/streaming_stft.rs

25 issues found:

- Line 167: `let pad_length = config.frame_length / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 310: `complex_spectrum.mapv(|c| c.norm().powf(self.config.power))`
  - **Fix**: Mathematical operation .powf( without validation
- Line 314: `magnitude_spectrum.mapv(|m| (m + self.config.log_epsilon).ln())`
  - **Fix**: Mathematical operation .ln() without validation
- Line 328: `self.config.frame_length / 2 + self.config.hop_length`
  - **Fix**: Division without zero check - use safe_divide()
- Line 336: `self.get_latency_samples() as f64 / sample_rate`
  - **Fix**: Division without zero check - use safe_divide()
- Line 357: `let pad_length = self.config.frame_length / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 401: `let frame_slice = windowed_frame.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 408: `let n_freq = self.config.frame_length / 2 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 423: `spectrum.mapv(|c| Complex64::new(c.norm().powf(self.config.power), 0.0))`
  - **Fix**: Mathematical operation .powf( without validation
- Line 428: `.mapv(|c| Complex64::new((c.re + self.config.log_epsilon).ln(), 0.0)))`
  - **Fix**: Mathematical operation .ln() without validation
- Line 590: `let stft = StreamingStft::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 605: `let mut stft = StreamingStft::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 613: `.map(|i| (2.0 * PI * freq * i as f64 / fs).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 619: `let result = stft.process_frame(&input_frame).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 622: `let spectrum = result.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 636: `let mut stft = StreamingStft::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 639: `let magnitude_result = stft.process_magnitude_frame(&input).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 642: `let magnitude_spectrum = magnitude_result.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 655: `let mut rt_stft = RealTimeStft::new(config, 128, 10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 658: `let new_spectra = rt_stft.process_block(&input_block).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 673: `let stft = StreamingStft::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 691: `let mut stft = StreamingStft::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 694: `let results = stft.process_batch(&input_data, 64).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 709: `let mut stft = StreamingStft::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 716: `let flushed_results = stft.flush().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/swt.rs

12 issues found:

- Line 110: `let offset = filter_len / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 133: `let scale_factor = 2.0_f64.sqrt().powi(level as i32);`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 215: `let scale_factor = 1.0 / 2.0_f64.sqrt().powi(level as i32);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 215: `let scale_factor = 1.0 / 2.0_f64.sqrt().powi(level as i32);`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 232: `let offset = filter_len / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 560: `let (approx, detail) = swt_decompose(&signal, Wavelet::Haar, 1, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 586: `let (approx, detail) = swt_decompose(&signal, Wavelet::Haar, 1, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 589: `let reconstructed = swt_reconstruct(&approx, &detail, Wavelet::Haar, 1).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 607: `let (approx2, detail2) = swt_decompose(&step_signal, Wavelet::Haar, 1, None).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 623: `let reconstructed2 = swt_reconstruct(&approx2, &detail2, Wavelet::Haar, 1).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 633: `let (details, approx) = swt(&signal, Wavelet::Haar, 2, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 650: `let reconstructed = iswt(&details, &approx, Wavelet::Haar).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/swt2d.rs

7 issues found:

- Line 600: `let level_weight = (total_levels - level_idx) as f64 / total_weight as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 629: `let result = swt2d_decompose(&image, Wavelet::Haar, 1, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 652: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 655: `let decomposition = swt2d_decompose(&data, Wavelet::Haar, 1, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 658: `let reconstructed = swt2d_reconstruct(&decomposition, Wavelet::Haar, 1, None).un...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 701: `let decompositions = swt2d(&image, Wavelet::Haar, levels, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 715: `let reconstructed = iswt2d(&decompositions, Wavelet::Haar, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/sysid.rs

43 issues found:

- Line 535: `let nfft = config.nfft.unwrap_or(next_power_of_2(input.len() / 8));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 544: `input.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 560: `output.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 572: `freq_response[i] = pxy[i] / pxx[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 575: `let coherence_val = pxy[i].norm_sqr() / (pxx[i].abs() * pyy[i]);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 612: `let window_norm = window_array.mapv(|w| w * w).sum().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 615: `let mut pxy_acc = Array1::<Complex64>::zeros(nfft / 2 + 1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 633: `nfft / 2`
  - **Fix**: Division without zero check - use safe_divide()
- Line 635: `(nfft - 1) / 2`
  - **Fix**: Division without zero check - use safe_divide()
- Line 652: `pxy_acc.mapv_inplace(|x| x / scale);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 655: `let freqs = Array1::linspace(0.0, fs / 2.0, nfft / 2 + 1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 681: `let mut freq_response = Array1::<Complex64>::zeros(nfft / 2 + 1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 682: `let mut coherence = Array1::<f64>::zeros(nfft / 2 + 1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 684: `for i in 0..=nfft / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 685: `let idx = if i == nfft / 2 { nfft / 2 } else { i };`
  - **Fix**: Division without zero check - use safe_divide()
- Line 687: `freq_response[i] = output_fft[idx] / input_fft[idx];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 693: `let freqs = Array1::linspace(0.0, fs / 2.0, nfft / 2 + 1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 722: `let nfft = config.nfft.unwrap_or(next_power_of_2(input.len() / 8));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 728: `output.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 738: `input.as_slice().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 753: `freq_response[i] = Complex64::new(pyy[i], 0.0) / pxy[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 755: `let coherence_val = pxy[i].norm_sqr() / (pxx[i] * pyy[i]);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 867: `let fit_percentage = 100.0 * (1.0 - error_sum / signal_sum).max(0.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 873: `error_variance: error_sum / n_freq as f64,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 935: `if ar_order + ma_order >= signal.len() / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 943: `let log_likelihood = -0.5 * n * (2.0 * PI * noise_var).ln() - 0.5 * n;`
  - **Fix**: Mathematical operation .ln() without validation
- Line 947: `OrderSelection::BIC => -2.0 * log_likelihood + k as f64 * n.ln(),`
  - **Fix**: Mathematical operation .ln() without validation
- Line 949: `-2.0 * log_likelihood + 2.0 * k as f64 * n / (n - k as f64 - 1.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1009: `1.0 - ss_res / ss_tot`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1018: `let log_likelihood = -0.5 * n * (2.0 * PI * mse).ln() - 0.5 * n;`
  - **Fix**: Mathematical operation .ln() without validation
- Line 1020: `let bic = -2.0 * log_likelihood + model_order as f64 * n.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 1023: `let fpe = mse * (n + model_order as f64) / (n - model_order as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1151: `let gain = &p_phi / denominator;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1164: `self.covariance = (&self.covariance - &k_phi_t_p) / self.forgetting_factor;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1190: `let fit = 1.0 - ss_res / ss_tot;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1218: `lb_stat += autocorr * autocorr / (n - lag) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1225: `(-lb_stat / 2.0).exp()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1287: `let angle = -2.0 * PI * (k * t) as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1338: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1356: `let result = identify_ar_model(&signal, 5, ARMethod::Burg, OrderSelection::AIC)....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1380: `let _ = rls.update(regression, *output).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1395: `let validation = validate_model(&predicted, &actual, 2, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1418: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/sysid_advanced.rs

26 issues found:

- Line 58: `let cost = new_residuals.mapv(|r| r * r).sum() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 176: `let cost = residuals.mapv(|r| r * r).sum() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 243: `(&new_d - &d).norm() + (&new_f - &f).norm()) / 4.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 249: `let cost = final_residuals.mapv(|r| r * r).sum() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 282: `let cost = final_residuals.mapv(|r| r * r).sum() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 323: `let cost = residuals.mapv(|r| r * r).sum() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 361: `let u_svd = u_svd.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 362: `let vt = vt.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 370: `let gamma = u1.dot(&s1.mapv(|x| x.sqrt()));`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 413: `let cost = residuals.mapv(|r| r * r).sum() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 489: `let variance = residuals.mapv(|r| r * r).sum() / n_samples as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 657: `let sensitivity = (&y_plus - y_sim) / h;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 661: `gradient[i] = 2.0 * error.dot(&sensitivity) / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 675: `let sensitivity2 = (&y_plus2 - y_sim) / h;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 677: `hessian[[i, j]] = 2.0 * sensitivity.dot(&sensitivity2) / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 996: `phi[[i, na + nb]] = 1.0 / (1.0 + (-scale * (x - offset)).exp());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1001: `let center = -2.0 + 4.0 * k as f64 / n_nonlinear as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1004: `phi[[i, na + nb + k]] = (-(x - center).powi(2) / (2.0 * width * width)).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1072: `let sigma2 = residuals.mapv(|r| r * r).sum() / (n - k) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1076: `let std_errors = Array1::from_shape_fn(k, |i| covariance[[i, i]].sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1099: `r[k] = sum / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1110: `let u = u.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1111: `let vt = vt.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1118: `s_inv[[i, i]] = 1.0 / s[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1179: `).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1193: `).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/sysid_enhanced.rs

27 issues found:

- Line 316: `let gain = p_phi / denominator;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 324: `self.covariance = (&self.covariance - &outer.dot(&self.covariance)) / self.lambd...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 343: `let na = self.phi_buffer.len() / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 363: `self.covariance.diag().map(|x| x.sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 377: `let input_mean = proc_input.mean().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 378: `let output_mean = proc_output.mean().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 418: `sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 419: `sorted[sorted.len() / 2]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 442: `(config.max_order / 2, config.max_order / 2, 1)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 462: `let sigma2 = residuals.dot(&residuals) / (n - na - nb) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 463: `let covariance = phi_t_phi.inv().unwrap() * sigma2;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 464: `let std_errors = covariance.diag().map(|x| x.sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 479: `let cost = residuals.dot(&residuals) / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 547: `let sigma2 = residuals.dot(&residuals) / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 550: `let aic = n as f64 * sigma2.ln() + 2.0 * k as f64;`
  - **Fix**: Mathematical operation .ln() without validation
- Line 631: `let y_mean = output.mean().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 637: `let fit_percentage = 100.0 * (1.0 - ss_res / ss_tot);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 642: `let sigma2 = ss_res / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 644: `let aic = n * sigma2.ln() + 2.0 * k;`
  - **Fix**: Mathematical operation .ln() without validation
- Line 645: `let bic = n * sigma2.ln() + k * n.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 646: `let fpe = sigma2 * (n + k) / (n - k);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 712: `let max_lag = 20.min(residuals.len() / 4);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 715: `let r_mean = residuals.mean().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 718: `.sum::<f64>() / residuals.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 725: `autocorrelation[lag] = sum / ((residuals.len() - lag) as f64 * r0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 766: `let error = sysid.update(input, output).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 792: `let result = enhanced_system_identification(&input, &output, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/tv.rs

37 issues found:

- Line 181: `p[i] = (p[i] + step * grad_div_p[i]) / (1.0 + step * grad_div_p[i].abs());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 194: `let relative_change = (change / norm.max(1e-10)).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 194: `let relative_change = (change / norm.max(1e-10)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 225: `let step_size = 1.0 / l;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 245: `t = (1.0 + f64::sqrt(1.0 + 4.0 * t_prev * t_prev)) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 248: `let momentum = (t_prev - 1.0) / t;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 263: `let relative_change = (change / norm.max(1e-10)).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 263: `let relative_change = (change / norm.max(1e-10)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 450: `/ (1.0 + step * grad_div_p1[[i, j]].abs());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 454: `/ (1.0 + step * grad_div_p2[[i, j]].abs());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 470: `let norm = f64::max(sum.sqrt(), 1.0);`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 472: `p1[[i, j]] = new_p1 / norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 473: `p2[[i, j]] = new_p2 / norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 491: `let relative_change = (change / norm.max(1e-10)).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 491: `let relative_change = (change / norm.max(1e-10)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 522: `let step_size = 1.0 / l;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 544: `t = (1.0 + f64::sqrt(1.0 + 4.0 * t_prev * t_prev)) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 547: `let momentum = (t_prev - 1.0) / t;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 566: `let relative_change = (change / norm.max(1e-10)).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 566: `let relative_change = (change / norm.max(1e-10)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 640: `.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 645: `.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 650: `* (dx_forward / forward_norm + dy_forward / forward_norm`
  - **Fix**: Division without zero check - use safe_divide()
- Line 651: `- dx_backward / backward_norm`
  - **Fix**: Division without zero check - use safe_divide()
- Line 652: `- dy_backward / backward_norm);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 819: `/ (1.0 + step * grad[[i, j, pc]].abs());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 834: `p[[i, j, pc]] = (p[[i, j, pc]] + step * grad[[i, j, pc]]) / norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 855: `let relative_change = (change / norm.max(1e-10)).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 855: `let relative_change = (change / norm.max(1e-10)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1014: `result[[i, j]] = sum / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1097: `/ (1.0 + step * grad1[[i, j]].abs());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1099: `/ (1.0 + step * grad2[[i, j]].abs());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1111: `let norm = f64::max(sum.sqrt(), 1.0);`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1113: `p1[[i, j]] = new_p1 / norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1114: `p2[[i, j]] = new_p2 / norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1136: `let relative_change = (change / norm.max(1e-10)).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1136: `let relative_change = (change / norm.max(1e-10)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation

### src/utilities/spectral.rs

46 issues found:

- Line 65: `let dt = 1.0 / fs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 125: `let normalized = psd_f64.iter().map(|&p| p / sum).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 212: `let centroid = weighted_sum / sum;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 306: `let spread = (weighted_sum_sq_diff / sum).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 306: `let spread = (weighted_sum_sq_diff / sum).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 416: `let skewness = weighted_sum_cubed_diff / (sum * spread_val.powi(3));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 529: `let kurtosis = weighted_sum_fourth_power_diff / (sum * spread_val.powi(4)) - 3.0...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 593: `let arithmetic_mean = positive_psd.iter().sum::<f64>() / positive_psd.len() as f...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 602: `let log_sum: f64 = positive_psd.iter().map(|&p| p.ln()).sum();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 603: `let geometric_mean = (log_sum / positive_psd.len() as f64).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 606: `let flatness = geometric_mean / arithmetic_mean;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 696: `let flux = (diff.iter().map(|&d| d * d).sum::<f64>()).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 859: `let mean_val = psd_f64.iter().sum::<f64>() / psd_f64.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 868: `let crest = max_val / mean_val;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 952: `weighted_sum += (psd_f64[i] - psd_f64[0]) / freqs_f64[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 957: `let decrease = weighted_sum / amplitude_sum;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1025: `let freq_mean = freqs_f64.iter().sum::<f64>() / freqs_f64.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1026: `let psd_mean = psd_f64.iter().sum::<f64>() / psd_f64.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1047: `let slope = covariance / variance;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1136: `let band_width = freq_range / n_bands as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1159: `band_psd.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1181: `let contrast = (peak / valley).log10();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1265: `let threshold_linear = peak_magnitude * 10.0_f64.powf(threshold_db / 10.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1265: `let threshold_linear = peak_magnitude * 10.0_f64.powf(threshold_db / 10.0);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 1275: `let t = (threshold_linear - psd_f64[i]) / (psd_f64[i + 1] - psd_f64[i]);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1288: `crossings.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1365: `.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1477: `peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1521: `let esd = energy_spectral_density(&psd, fs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1525: `assert_relative_eq!(esd[i], p / fs, epsilon = 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1534: `let norm_psd = normalized_psd(&psd).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1543: `assert_relative_eq!(norm_psd[i] / norm_psd[0], p / psd[0], epsilon = 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1554: `let centroid = spectral_centroid(&psd, &freqs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1563: `let centroid = spectral_centroid(&psd, &freqs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1575: `let spread = spectral_spread(&psd, &freqs, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1584: `let spread = spectral_spread(&psd, &freqs, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1595: `let flatness = spectral_flatness(&psd).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1603: `let flatness = spectral_flatness(&psd).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1615: `let flux_l1 = spectral_flux(&psd1, &psd2, "l1").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1616: `let flux_l2 = spectral_flux(&psd1, &psd2, "l2").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1617: `let flux_max = spectral_flux(&psd1, &psd2, "max").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1628: `let flux_l1 = spectral_flux(&psd1, &psd2, "l1").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1629: `let flux_l2 = spectral_flux(&psd1, &psd2, "l2").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1630: `let flux_max = spectral_flux(&psd1, &psd2, "max").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1644: `let rolloff = spectral_rolloff(&psd, &freqs, 0.95).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1653: `let rolloff = spectral_rolloff(&psd, &freqs, 0.95).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/utilities/tests.rs

41 issues found:

- Line 17: `let esd = energy_spectral_density(&psd, fs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 21: `assert_relative_eq!(esd[i], p / fs, epsilon = 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 30: `let norm_psd = normalized_psd(&psd).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 39: `assert_relative_eq!(norm_psd[i] / norm_psd[0], p / psd[0], epsilon = 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 50: `let centroid = spectral_centroid(&psd, &freqs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 59: `let centroid = spectral_centroid(&psd, &freqs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 71: `let spread = spectral_spread(&psd, &freqs, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 80: `let spread = spectral_spread(&psd, &freqs, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 92: `let skewness = spectral_skewness(&psd, &freqs, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 101: `let skewness = spectral_skewness(&psd, &freqs, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 117: `let skewness = spectral_skewness(&psd, &freqs, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 136: `let kurtosis = spectral_kurtosis(&psd, &freqs, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 145: `let kurtosis = spectral_kurtosis(&psd, &freqs, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 156: `let flatness = spectral_flatness(&psd).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 164: `let flatness = spectral_flatness(&psd).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 176: `let flux_l1 = spectral_flux(&psd1, &psd2, "l1").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 177: `let flux_l2 = spectral_flux(&psd1, &psd2, "l2").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 178: `let flux_max = spectral_flux(&psd1, &psd2, "max").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 189: `let flux_l1 = spectral_flux(&psd1, &psd2, "l1").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 190: `let flux_l2 = spectral_flux(&psd1, &psd2, "l2").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 191: `let flux_max = spectral_flux(&psd1, &psd2, "max").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 209: `let rolloff_95 = spectral_rolloff(&psd, &freqs, 0.95).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 210: `let rolloff_50 = spectral_rolloff(&psd, &freqs, 0.50).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 222: `let rolloff = spectral_rolloff(&psd, &freqs, 0.95).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 234: `let crest = spectral_crest(&psd).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 242: `let crest = spectral_crest(&psd).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 255: `let decrease = spectral_decrease(&psd, &freqs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 264: `let decrease = spectral_decrease(&psd, &freqs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 276: `let slope = spectral_slope(&psd, &freqs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 285: `let slope = spectral_slope(&psd, &freqs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 297: `let contrast = spectral_contrast(&psd, &freqs, 2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 311: `let contrast = spectral_contrast(&psd, &freqs, 2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 325: `let bandwidth_3db = spectral_bandwidth(&psd, &freqs, -3.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 326: `let bandwidth_6db = spectral_bandwidth(&psd, &freqs, -6.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 342: `let (dominant_freq, magnitude) = dominant_frequency(&psd, &freqs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 355: `let peaks = dominant_frequencies(&psd, &freqs, 3, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 408: `let centroid = spectral_centroid(&psd.to_vec(), &freqs.to_vec()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 413: `Array2::from_shape_vec((1, 7), vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]).unwrap()...`
  - **Fix**: Handle array creation errors properly
- Line 417: `let spread = spectral_spread(&psd_slice_vec, &freqs.to_vec(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 421: `let crest = spectral_crest(&psd.to_vec()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 424: `let (dominant_freq, _) = dominant_frequency(&psd.to_vec(), &freqs.to_vec()).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/utils.rs

20 issues found:

- Line 73: `let pad_before = pad_length / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 125: `let t = (i + 1) as f64 / (pad_before + 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 133: `let t = (i + 1) as f64 / (pad_after + 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 163: `let mean_val = x_f64.iter().sum::<f64>() / x_f64.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 285: `let scale = 1.0 / sum_of_squares.sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 285: `let scale = 1.0 / sum_of_squares.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 300: `let scale = 1.0 / peak;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 366: `let padded = zero_pad(&signal, 10, "constant", Some(0.0)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 378: `let padded = zero_pad(&signal, 8, "constant", Some(5.0)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 394: `let padded = zero_pad(&signal, 8, "edge", None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 404: `let padded = zero_pad(&signal, 8, "mean", None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 414: `let window = get_window("hamming", 10, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 420: `let middle_index = window.len() / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 434: `let window = get_window("hann", 10, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 440: `let middle_index = window.len() / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 455: `let normalized = normalize(&signal, "energy").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 462: `assert_relative_eq!(normalized[1] / normalized[0], 2.0, epsilon = 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 463: `assert_relative_eq!(normalized[2] / normalized[0], 3.0, epsilon = 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 464: `assert_relative_eq!(normalized[3] / normalized[0], 4.0, epsilon = 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 471: `let normalized = normalize(&signal, "peak").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/waveforms.rs

35 issues found:

- Line 61: `let phi_rad = phi * PI / 180.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 82: `let beta = (f1 - f0) / t1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 87: `let beta = (f1 - f0) / (t1 * t1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 88: `2.0 * PI * (f0 * t + beta * t * t * t / 3.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 92: `let k = (f1 / f0).powf(1.0 / t1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 92: `let k = (f1 / f0).powf(1.0 / t1);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 97: `2.0 * PI * f0 * (k.powf(t) - 1.0) / (k - 1.0).ln()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 97: `2.0 * PI * f0 * (k.powf(t) - 1.0) / (k - 1.0).ln()`
  - **Fix**: Mathematical operation .ln() without validation
- Line 97: `2.0 * PI * f0 * (k.powf(t) - 1.0) / (k - 1.0).ln()`
  - **Fix**: Mathematical operation .powf( without validation
- Line 102: `let c = f0 * t1 * ((f1 / f0) - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 103: `2.0 * PI * c * (f0 * t / (f0 * t + c)).ln()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 103: `2.0 * PI * c * (f0 * t / (f0 * t + c)).ln()`
  - **Fix**: Mathematical operation .ln() without validation
- Line 181: `2.0 * t_cycle / width - 1.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 184: `-2.0 * (t_cycle - width) / (1.0 - width) + 1.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 358: `let signal = chirp(&t, 1.0, 1.0, 10.0, "linear", 0.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 375: `let signal = sawtooth(&t, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 391: `let signal = square(&t, 0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 401: `let signal = square(&t, 0.25).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 417: `let signal = gausspulse(&t, 1.0, 0.5, None, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 727: `let k = (f2 / f1).ln() / length; // Exponential rate constant`
  - **Fix**: Mathematical operation .ln() without validation
- Line 742: `let phase = 2.0 * PI * f1 * ((k * time).exp() - 1.0) / k;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 786: `let t: Vec<f64> = (0..num_samples).map(|i| i as f64 / sample_rate).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 791: `let rate = (f2 - f1) / duration;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 971: `let mls = mls_sequence(4, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 980: `let mls_custom = mls_sequence(5, Some(&[3, 5]), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 996: `let prbs = prbs_sequence(100, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1007: `let pink = pink_noise(1000, Some(42)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1018: `let brown = brown_noise(1000, Some(42)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1030: `let sweep = exponential_sweep(&t, 20.0, 20000.0, 0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1042: `let (t, sweep) = synchronized_sweep(44100.0, 1.0, 20.0, 20000.0, "linear").unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1049: `synchronized_sweep(44100.0, 1.0, 20.0, 20000.0, "logarithmic").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1055: `let ruler = golomb_ruler(4, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1067: `let seq = perfect_binary_sequence(7).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1081: `let pink1 = pink_noise(100, Some(123)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1082: `let pink2 = pink_noise(100, Some(123)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/wavelet_vis.rs

26 issues found:

- Line 354: `100.0 * energy_approx / total_energy`
  - **Fix**: Division without zero check - use safe_divide()
- Line 360: `100.0 * energy_detail / total_energy`
  - **Fix**: Division without zero check - use safe_divide()
- Line 429: `100.0 * energy_approx / total_energy`
  - **Fix**: Division without zero check - use safe_divide()
- Line 435: `100.0 * energy_detail / total_energy`
  - **Fix**: Division without zero check - use safe_divide()
- Line 494: `100.0 * energy_approx / total_energy`
  - **Fix**: Division without zero check - use safe_divide()
- Line 500: `100.0 * energy_detail / total_energy`
  - **Fix**: Division without zero check - use safe_divide()
- Line 578: `coefficients.mapv(|x| min_val + range_size * (x - coeff_min) / coeff_range)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 592: `let scale = max_val / max_abs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 596: `let half_range = range_size / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 598: `coefficients.mapv(|x| mid_point + half_range * x / max_abs)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 614: `let log_val = (1.0 + x.abs() / max_abs).ln();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 614: `let log_val = (1.0 + x.abs() / max_abs).ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 616: `let scaled = min_val + range_size * log_val / (2.0_f64.ln());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 616: `let scaled = min_val + range_size * log_val / (2.0_f64.ln());`
  - **Fix**: Mathematical operation .ln() without validation
- Line 631: `let lower_idx = (n as f64 * lower / 100.0).floor() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 632: `let upper_idx = (n as f64 * upper / 100.0).ceil() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 655: `min_val + range_size * (clipped - lower_val) / val_range`
  - **Fix**: Division without zero check - use safe_divide()
- Line 747: `let percent = 100.0 * (total_count as f64) / (total_coeffs as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 955: `let decomp = dwt2d_decompose(&image, Wavelet::Haar, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 985: `let decomps = wavedec2(&image, Wavelet::Haar, 1, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 988: `let arranged = arrange_multilevel_coefficients_2d(&decomps).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1010: `let decomp = dwt2d_decompose(&image, Wavelet::Haar, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1019: `let h_energy = energy.horizontal.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1020: `let v_energy = energy.vertical.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1021: `let d_energy = energy.diagonal.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1032: `let decomp = swt2d_decompose(&image, Wavelet::Haar, 1, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/wavelets/complex_wavelets.rs

35 issues found:

- Line 44: `let mid_point = (points - 1) as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 48: `let t = (i as f64 - mid_point) / s;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 57: `let norm = (PI * s * s).sqrt().recip();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 117: `let mid_point = (points - 1) as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 121: `let norm = 1.0 / (PI * bandwidth * bandwidth * scale * scale).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 121: `let norm = 1.0 / (PI * bandwidth * bandwidth * scale * scale).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 124: `let correction = (-center_frequency * center_frequency / (2.0 * bandwidth * band...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 127: `let t = (i as f64 - mid_point) / scale;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 131: `Complex64::new(symmetry * t * t / 2.0, 0.0).exp()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 140: `let gauss = (-t * t / (2.0 * bandwidth * bandwidth)).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 201: `let norm = (2.0_f64.powf(m) * fact_2m_1 / (std::f64::consts::PI * fact_m)).sqrt(...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 201: `let norm = (2.0_f64.powf(m) * fact_2m_1 / (std::f64::consts::PI * fact_m)).sqrt(...`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 201: `let norm = (2.0_f64.powf(m) * fact_2m_1 / (std::f64::consts::PI * fact_m)).sqrt(...`
  - **Fix**: Mathematical operation .powf( without validation
- Line 204: `let mid_point = (points - 1) as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 208: `let t = (i as f64 - mid_point) / scale;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 212: `let factor = Complex64::new(0.0, 1.0).powf(order as f64);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 213: `let denom = (1.0 - Complex64::new(0.0, t)).powf(order as f64 + 1.0);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 214: `norm * factor / denom`
  - **Fix**: Division without zero check - use safe_divide()
- Line 217: `let val = norm * 2.0_f64.powf(m - 1.0) * fact_2m_1 / (fact_m * (2.0 * m - 1.0));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 217: `let val = norm * 2.0_f64.powf(m - 1.0) * fact_2m_1 / (fact_m * (2.0 * m - 1.0));`
  - **Fix**: Mathematical operation .powf( without validation
- Line 270: `let mid_point = (points - 1) as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 276: `let gauss = (-t * t / 2.0).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 292: `let factor = Complex64::new(0.0, -1.0).powf(m as f64) / (factorial(m) as f64).sq...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 292: `let factor = Complex64::new(0.0, -1.0).powf(m as f64) / (factorial(m) as f64).sq...`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 292: `let factor = Complex64::new(0.0, -1.0).powf(m as f64) / (factorial(m) as f64).sq...`
  - **Fix**: Mathematical operation .powf( without validation
- Line 300: `let t = (i as f64 - mid_point) / scale;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 306: `let normalization = (PI * scale * scale).sqrt().recip();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 367: `let norm = scale.sqrt().recip() * (2.0 * bandwidth).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 370: `let mid_point = (points - 1) as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 374: `let t = (i as f64 - mid_point) / scale;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 381: `(bandwidth * PI * t).sin() / (bandwidth * PI * t)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 461: `let mid_point = (points - 1) as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 466: `let norm = scale.sqrt().recip() * (bandwidth * (order as f64)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 469: `let t = (i as f64 - mid_point) / scale;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 479: `let sinc_term = (omega / 2.0).sin() / (omega / 2.0);`
  - **Fix**: Division without zero check - use safe_divide()

### src/wavelets/cwt.rs

2 issues found:

- Line 34: `let start = (nh - 1) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 67: `let start = (nh - 1) / 2;`
  - **Fix**: Division without zero check - use safe_divide()

### src/wavelets/dual_tree_complex.rs

53 issues found:

- Line 223: `.map(|(&a, &b)| (a + b) / 2.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 325: `(ya[[i, j]] + yb[[i, j]]) / 2.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 374: `let mut ya_row_filtered = Array3::zeros((rows, cols / 2, 2));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 375: `let mut yb_row_filtered = Array3::zeros((rows, cols / 2, 2));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 386: `for j in 0..cols / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 395: `let mut ya_subbands = Array3::zeros((rows / 2, cols / 2, 4));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 396: `let mut yb_subbands = Array3::zeros((rows / 2, cols / 2, 4));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 398: `for j in 0..cols / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 408: `for i in 0..rows / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 474: `let real = (ya_subbands[[i, j, 1]] + ya_subbands[[i, j, 2]]) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 475: `let imag = (yb_subbands[[i, j, 1]] + yb_subbands[[i, j, 2]]) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 483: `let real = (ya_subbands[[i, j, 1]] - ya_subbands[[i, j, 2]]) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 484: `let imag = (yb_subbands[[i, j, 1]] - yb_subbands[[i, j, 2]]) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 510: `let real = (ya_subbands[[i, j, 2]] + ya_subbands[[i, j, 3]]) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 511: `let imag = (yb_subbands[[i, j, 2]] + yb_subbands[[i, j, 3]]) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 519: `let real = (ya_subbands[[i, j, 2]] - ya_subbands[[i, j, 3]]) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 520: `let imag = (yb_subbands[[i, j, 2]] - yb_subbands[[i, j, 3]]) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 593: `let mut result = Vec::with_capacity(conv_len / 2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 650: `extended.extend_from_slice(signal.as_slice().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 669: `extended.extend_from_slice(signal.as_slice().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 715: `h0a.mapv_inplace(|x| x / h0a_sum);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 746: `h0b.mapv_inplace(|x| x / h0b_sum);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 781: `-1.0 / 8.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 782: `1.0 / 4.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 783: `3.0 / 4.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 784: `1.0 / 4.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 785: `-1.0 / 8.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 787: `let h1a = Array1::from_vec(vec![1.0 / 2.0, -1.0, 1.0 / 2.0]);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 791: `-1.0 / 16.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 792: `1.0 / 8.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 793: `5.0 / 8.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 794: `5.0 / 8.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 795: `1.0 / 8.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 796: `-1.0 / 16.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 798: `let h1b = Array1::from_vec(vec![1.0 / 4.0, -1.0 / 2.0, 1.0 / 2.0, -1.0 / 4.0]);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 841: `let processor = DtcwtProcessor::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 846: `.map(|i| (2.0 * PI * i as f64 / 16.0).sin() + 0.5 * (2.0 * PI * i as f64 / 8.0)....`
  - **Fix**: Division without zero check - use safe_divide()
- Line 850: `let dtcwt_result = processor.dtcwt_1d_forward(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 855: `let reconstructed = processor.dtcwt_1d_inverse(&dtcwt_result).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 863: `/ n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 874: `let processor = DtcwtProcessor::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 879: `Array2::from_shape_fn((rows, cols), |(i, j)| ((i as f64 + j as f64) / 8.0).sin()...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 882: `let dtcwt_result = processor.dtcwt_2d_forward(&image).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 899: `let processor = DtcwtProcessor::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 903: `let signal: Array1<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / 8.0).sin()).coll...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 912: `let dtcwt1 = processor.dtcwt_1d_forward(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 913: `let dtcwt2 = processor.dtcwt_1d_forward(&shifted_signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 932: `/ (mag1.iter().map(|x| x * x).sum::<f64>().sqrt()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 932: `/ (mag1.iter().map(|x| x * x).sum::<f64>().sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 933: `* mag2.iter().map(|x| x * x).sum::<f64>().sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 946: `let filters = create_dtcwt_filters(FilterSet::Kingsbury).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 968: `let processor = DtcwtProcessor::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 971: `let extended = processor.extend_signal(&signal, 6).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/wavelets/real_wavelets.rs

6 issues found:

- Line 38: `let amplitude = 2.0 / (std::f64::consts::PI.powf(0.25) * (3.0 * a).sqrt());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 38: `let amplitude = 2.0 / (std::f64::consts::PI.powf(0.25) * (3.0 * a).sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 38: `let amplitude = 2.0 / (std::f64::consts::PI.powf(0.25) * (3.0 * a).sqrt());`
  - **Fix**: Mathematical operation .powf( without validation
- Line 42: `let mid_point = (points - 1) as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 48: `let mod_term = 1.0 - xsq / wsq;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 49: `let gauss = (-xsq / (2.0 * wsq)).exp();`
  - **Fix**: Division without zero check - use safe_divide()

### src/wavelets/scalogram.rs

1 issues found:

- Line 320: `.map(|&s| central_freq / (s * sampling_period))`
  - **Fix**: Division without zero check - use safe_divide()

### src/wavelets/tests.rs

7 issues found:

- Line 11: `let wavelet = morlet(points, 6.0, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 17: `let mid_point = (points - 1) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 32: `let freqs = scale_to_frequency(&scales, central_freq, dt).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 44: `let expected_freq = central_freq / (scale * dt);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 53: `let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / 32.0).sin()).collec...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 62: `let coeffs = transform::cwt(&signal, wavelet_fn, &scales).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 69: `let magnitude = cwt_magnitude(&signal, wavelet_fn, &scales, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/wiener.rs

28 issues found:

- Line 174: `let wiener_gain = power / (power + snr_factor * noise_power + config.regularizat...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 222: `let half_window = config.window_size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 270: `/ window.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 281: `max_var / (local_var + config.regularization)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 331: `let snr = signal_power / noise_power;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 387: `let half_h = win_size[0] / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 388: `let half_w = win_size[1] / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 408: `/ (window.len() as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 413: `max_var / (local_var + 1e-10)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 486: `.take(n / 2 + 1)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 487: `.map(|c| c.norm_sqr() / n as f64),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 504: `for i in 0..=n / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 520: `.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 526: `if i > 0 && i < n / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 584: `let half_n = n / 2 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 587: `psd_est[i] = fft_result[i].norm_sqr() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 603: `Array1::from_elem(n / 2 + 1, noise_var)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 611: `for i in 0..=n / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 623: `s_power / (s_power + n_power)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 635: `if i > 0 && i < n / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 697: `let k_gain = p_pred / (p_pred + measurement_var);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 715: `(values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 717: `values[values.len() / 2]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 728: `(deviations[deviations.len() / 2 - 1] + deviations[deviations.len() / 2]) / 2.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 730: `deviations[deviations.len() / 2]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 747: `let power = signal.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / signal.len...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 781: `let half_window = window_size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 790: `smoothed[i] = sum / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()

### src/window/kaiser.rs

15 issues found:

- Line 50: `let k = 2.0 * i as f64 / (n - 1) as f64 - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 55: `i0(beta * term.sqrt()) / i0_beta`
  - **Fix**: Division without zero check - use safe_divide()
- Line 55: `i0(beta * term.sqrt()) / i0_beta`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 102: `let _len_half = n / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 113: `for j in 1..(n / 2) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 116: `let angle = 2.0 * PI * i as f64 * j as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 122: `if n / 2 < kaiser_win.len() {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 124: `sum += kaiser_win[n / 2] * angle.cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 128: `(sum / n as f64).sqrt()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 128: `(sum / n as f64).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 140: `let angle = 2.0 * PI * i as f64 * j as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 146: `(sum / n as f64).sqrt()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 146: `(sum / n as f64).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 163: `let window = kaiser(10, 5.0, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 179: `let window = kaiser_bessel_derived(10, 5.0, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/window/mod.rs

100 issues found:

- Line 140: `let w_val = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 177: `let w_val = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 215: `let w_val = 0.42 - 0.5 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 216: `+ 0.08 * (4.0 * PI * i as f64 / (n - 1) as f64).cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 252: `let m2 = (n - 1) as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 254: `let w_val = 1.0 - ((i as f64 - m2) / m2).abs();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 288: `let m2 = (n as f64 - 1.0) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 290: `let w_val = 1.0 - ((i as f64 - m2) / (m2 + 1.0)).abs();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 334: `let w_val = a0 - a1 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 335: `+ a2 * (4.0 * PI * i as f64 / (n - 1) as f64).cos()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 336: `- a3 * (6.0 * PI * i as f64 / (n - 1) as f64).cos()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 337: `+ a4 * (8.0 * PI * i as f64 / (n - 1) as f64).cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 406: `let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 451: `let x = 2.0 * i as f64 / n1 - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 503: `let w_val = a0 - a1 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 504: `+ a2 * (4.0 * PI * i as f64 / (n - 1) as f64).cos()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 505: `- a3 * (6.0 * PI * i as f64 / (n - 1) as f64).cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 548: `let w_val = a0 - a1 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 549: `+ a2 * (4.0 * PI * i as f64 / (n - 1) as f64).cos()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 550: `- a3 * (6.0 * PI * i as f64 / (n - 1) as f64).cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 587: `let w_val = (PI * i as f64 / (n - 1) as f64).sin();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 622: `let center_val = center.unwrap_or(((n - 1) as f64) / 2.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 626: `let w_val = (-((i as f64 - center_val).abs() / tau)).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 675: `let width = (alpha * (n - 1) as f64 / 2.0).floor() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 680: `0.5 * (1.0 + (PI * (-1.0 + 2.0 * i as f64 / (alpha * (n - 1) as f64))).cos())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 683: `+ (PI * (-2.0 / alpha + 1.0 + 2.0 * i as f64 / (alpha * (n - 1) as f64))).cos())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 723: `let fac = (i as f64) / (n - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 770: `if nw <= 0.0 || nw >= m as f64 / 2.0 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 786: `let omega = 2.0 * PI * nw / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 792: `let k = i as f64 - (n as f64 - 1.0) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 799: `*off_diag_val = k * (n as f64 - k) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 807: `sorted_indices.sort_by(|&a, &b| eigenvals[b].partial_cmp(&eigenvals[a]).unwrap()...`
  - **Fix**: Use .get() with proper bounds checking
- Line 821: `let norm = window.iter().map(|&x| x * x).sum::<f64>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 911: `let norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 953: `let norm = new_v.iter().map(|&x| x * x).sum::<f64>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 983: `let window = hamming(10, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1000: `let window = hann(10, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1017: `let window = blackman(10, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1028: `let window = bartlett(10, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1051: `let window = flattop(10, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1062: `let window = boxcar(10, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1073: `let window = bohman(10, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1088: `let window = triang(10, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1099: `let window = dpss(64, 2.5, None, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1103: `let norm = window.iter().map(|&x| x * x).sum::<f64>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1119: `let windows = dpss_windows(32, 2.0, Some(3), true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1125: `let norm = window.iter().map(|&x| x * x).sum::<f64>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1205: `i as f64 - (m - 1) as f64 / 2.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1208: `i as f64 - m as f64 / 2.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1211: `let x = n / a as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1218: `let sinc_x = (PI * x).sin() / (PI * x);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1219: `let x_a = x / a as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1223: `(PI * x_a).sin() / (PI * x_a)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1244: `let window = lanczos(10, 2, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1268: `let window = lanczos(8, 2, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1280: `let window_a2 = lanczos(20, 2, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1281: `let window_a3 = lanczos(20, 3, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1295: `let window1 = lanczos(1, 2, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1298: `let window2 = lanczos(2, 2, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1372: `let nenbw = n as f64 * power_gain / (coherent_gain * coherent_gain);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1388: `let bin_05_idx = fft_len / (2 * n);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1393: `let frac_idx = 0.5 * fft_len as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1467: `0.5842 * (sidelobe_db.abs() - 21.0).powf(0.4) + 0.07886 * (sidelobe_db.abs() - 2...`
  - **Fix**: Mathematical operation .powf( without validation
- Line 1494: `let mut magnitude = vec![0.0; n / 2 + 1];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1502: `let angle = -2.0 * PI * k as f64 * i as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1507: `*mag = (real * real + imag * imag).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1523: `for i in 1..freq_response.len().min(fft_len / window_len * 4) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1553: `let bandwidth_bins = 2.0 * right_point * window_len as f64 / fft_len as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1564: `let window = hann(64, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1565: `let analysis = analyze_window(&window, Some(1024)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1577: `let hann_win = hann(32, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1578: `let hamming_win = hamming(32, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1585: `let comparison = compare_windows(&windows).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1592: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1597: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1605: `let window1 = design_window_with_constraints(64, -10.0, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1606: `let window2 = design_window_with_constraints(64, -25.0, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1607: `let window3 = design_window_with_constraints(64, -60.0, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1622: `let window = design_optimal_kaiser(64, -60.0, 0.1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1626: `let hann_win = super::hann(32, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1627: `let analysis = analyze_window_transition(&hann_win, 0.3).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1635: `let hann_win = super::hann(32, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1636: `let hamming_win = super::hamming(32, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1642: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1691: `0.5842 * (atten - 21.0).powf(0.4) + 0.07886 * (atten - 21.0)`
  - **Fix**: Mathematical operation .powf( without validation
- Line 1763: `.take(freq_response.len() / 2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1775: `.take(freq_response.len() / 2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1784: `let freq_bin_width = 1.0 / fft_size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1786: `let transition_center = (upper_point + lower_point) as f64 * freq_bin_width / 2....`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1789: `let upper_db = 20.0 * (freq_response[upper_point] / peak_value).log10();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1790: `let lower_db = 20.0 * (freq_response[lower_point] / peak_value).log10();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1792: `(upper_db - lower_db) / transition_width`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1798: `let threshold_3db = peak_value / 2.0_f64.sqrt(); // -3dB`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1863: `(sidelobe_score + bandwidth_score) / 2.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1916: `let half_len = length / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1921: `(length - 1 - i) as f64 / (length - 1) as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1923: `i as f64 / (length - 1) as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1940: `.take(freq_response.len() / 2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1963: `let t = (x - x1) / (x2 - x1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1968: `control_points.last().unwrap().1`
  - **Fix**: Replace with ? operator or .ok_or()

### src/wpt.rs

13 issues found:

- Line 89: `Some(self.position / 2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 157: `left.parent_position().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 313: `let level = *nodes_by_level.keys().next().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 314: `let positions = nodes_by_level.get(&level).unwrap();`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 575: `let (left, right) = root.decompose().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 602: `let parent = WaveletPacket::reconstruct(&left, &right).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 623: `let tree = wp_decompose(&signal, Wavelet::Haar, 2, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 638: `let root = tree.get_node(0, 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 648: `let tree = wp_decompose(&signal, Wavelet::Haar, 1, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 655: `let reconstructed = reconstruct_from_nodes(&tree, &nodes).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 674: `let tree = wp_decompose(&signal, Wavelet::Haar, 1, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 680: `let approx_node = tree.get_node(1, 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 685: `let approx_only = reconstruct_from_nodes(&tree, &nodes).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/wpt2d.rs

16 issues found:

- Line 492: `let out_rows = rows / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 493: `let out_cols = cols / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 555: `let out_len = n / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 566: `let ext_idx = idx as isize - (filter_len as isize / 2) + j as isize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 576: `((idx as isize - (filter_len as isize / 2) + j as isize) % n as isize) as usize`
  - **Fix**: Division without zero check - use safe_divide()
- Line 579: `let ext_idx = idx as isize - (filter_len as isize / 2) + j as isize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 852: `let decomp = wpt2d_full(&image, Wavelet::Haar, 2, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 890: `let decomp = wpt2d_selective(&image, Wavelet::Haar, 3, ll_only_criterion, None)....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 921: `let decomp = wpt2d_full(&image, Wavelet::Haar, 2, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 924: `assert_eq!(decomp.get_packet(0, 0, 0).unwrap().path, "");`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 927: `assert_eq!(decomp.get_packet(1, 0, 0).unwrap().path, "LL");`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 928: `assert_eq!(decomp.get_packet(1, 0, 1).unwrap().path, "LH");`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 929: `assert_eq!(decomp.get_packet(1, 1, 0).unwrap().path, "HL");`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 930: `assert_eq!(decomp.get_packet(1, 1, 1).unwrap().path, "HH");`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 934: `assert_eq!(decomp.get_packet(2, 0, 0).unwrap().path, "LL-LL");`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 935: `assert_eq!(decomp.get_packet(2, 3, 3).unwrap().path, "HH-HH");`
  - **Fix**: Replace with ? operator or .ok_or()

### src/wpt_validation.rs

16 issues found:

- Line 57: `.map(|&x| NumCast::from(x).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 71: `let energy_ratio = tree_energy / input_energy;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 170: `let error_view = ArrayView1::from_shape(n, &mut errors).unwrap();`
  - **Fix**: Handle array creation errors properly
- Line 176: `let mean_error = errors.iter().map(|&e| e.abs()).sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 179: `let signal_power = compute_energy(original) / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 180: `let noise_power = compute_energy(&errors) / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 182: `10.0 * (signal_power / noise_power).log10()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 200: `Ok(coeffs_energy / signal_energy)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 229: `let mean = reconstructed.iter().sum::<f64>() / reconstructed.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 239: `impulse[signal.len() / 2] = 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 277: `Ok(passed_tests as f64 / total_tests as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 366: `let diff = (node_energy - dwt_energy).abs() / node_energy.max(1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 422: `let result = validate_wpt(&signal, Wavelet::Haar, 2, 1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 432: `let tree = wpt_decompose(&signal, Wavelet::DB(4), 3, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 433: `let ratio = validate_parseval_frame(&tree, &signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 441: `let score = test_numerical_stability(&signal, Wavelet::Haar, 2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/wvd.rs

23 issues found:

- Line 82: `Array1::from(hilbert::hilbert(signal.as_slice().unwrap())?)`
  - **Fix**: Handle array creation errors properly
- Line 138: `Array1::from(hilbert::hilbert(signal1.as_slice().unwrap())?)`
  - **Fix**: Handle array creation errors properly
- Line 144: `Array1::from(hilbert::hilbert(signal2.as_slice().unwrap())?)`
  - **Fix**: Handle array creation errors properly
- Line 204: `Array1::from(hilbert::hilbert(signal.as_slice().unwrap())?)`
  - **Fix**: Handle array creation errors properly
- Line 243: `let mut wvd = Array2::<Complex64>::zeros((n_fft / 2 + 1, n));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 251: `let half = w_len / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 267: `let offset = (n_fft - w.len()) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 274: `let offset = (w.len() - n_fft) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 288: `Some(w) => w.len() / 2,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 331: `scirs2_fft::fft(acorr.as_slice().unwrap(), None).expect("FFT computation failed"...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 334: `let n_freqs = n_fft / 2 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 354: `Array1::linspace(0.0, fs / 2.0, n_freqs)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 368: `let dt = 1.0 / fs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 453: `/ a.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 466: `/ b.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 499: `let wvd = wigner_ville(&signal, config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 513: `let first_quarter = n / 4;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 514: `let last_quarter = 3 * n / 4;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 522: `/ first_quarter as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 529: `/ (n - last_quarter) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 553: `let xwvd = cross_wigner_ville(&signal1, &signal2, config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 597: `let wvd = wigner_ville(&signal, config.clone()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 599: `smoothed_pseudo_wigner_ville(&signal, &time_window, &freq_window, config).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/dwt_boundary_test.rs

6 issues found:

- Line 11: `let extended = extend_signal(&signal, 4, "symmetric").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 31: `let extended = extend_signal(&signal, 4, "periodic").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 51: `let extended = extend_signal(&signal, 4, "zero").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 71: `let extended = extend_signal(&signal, 4, "constant").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 91: `let extended = extend_signal(&signal, 4, "reflect").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 121: `let extended = extend_signal(&signal, 4, "symmetric").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/dwt_daubechies_test.rs

13 issues found:

- Line 10: `let filters = wavelet.filters().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 33: `let (approx, detail) = dwt_decompose(&signal, wavelet, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 40: `let reconstructed = dwt_reconstruct(&approx, &detail, wavelet).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 55: `let coeffs = wavedec(&signal, wavelet, Some(3), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 61: `let reconstructed = waverec(&coeffs, wavelet).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 76: `let (approx, detail) = dwt_decompose(&signal, wavelet, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 77: `let reconstructed = dwt_reconstruct(&approx, &detail, wavelet).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 83: `let coeffs = wavedec(&signal, wavelet, Some(2), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 84: `let reconstructed_ml = waverec(&coeffs, wavelet).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 101: `let (approx, detail) = dwt_decompose(&signal, wavelet, Some(mode)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 102: `let reconstructed = dwt_reconstruct(&approx, &detail, wavelet).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 108: `let coeffs = wavedec(&signal, wavelet, Some(2), Some(mode)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 109: `let reconstructed_ml = waverec(&coeffs, wavelet).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/dwt_haar_test.rs

12 issues found:

- Line 9: `let filters = wavelet.filters().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 30: `let (approx, detail) = dwt_decompose(&signal, Wavelet::Haar, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 49: `let (approx, detail) = dwt_decompose(&signal, Wavelet::Haar, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 52: `let reconstructed = dwt_reconstruct(&approx, &detail, Wavelet::Haar).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 67: `let coeffs = wavedec(&signal, Wavelet::Haar, Some(2), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 73: `let reconstructed = waverec(&coeffs, Wavelet::Haar).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 86: `dwt_decompose(&signal, Wavelet::Haar, Some("symmetric")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 88: `dwt_decompose(&signal, Wavelet::Haar, Some("periodic")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 90: `dwt_decompose(&signal, Wavelet::Haar, Some("zero")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 101: `let recon_sym = dwt_reconstruct(&approx_sym, &detail_sym, Wavelet::Haar).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 102: `let recon_per = dwt_reconstruct(&approx_per, &detail_per, Wavelet::Haar).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 103: `let recon_zero = dwt_reconstruct(&approx_zero, &detail_zero, Wavelet::Haar).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/dwt_test.rs

4 issues found:

- Line 10: `let (approx, detail) = dwt_decompose(&signal, Wavelet::Haar, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 21: `let reconstructed = dwt_reconstruct(&approx, &detail, Wavelet::Haar).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 35: `let coeffs = wavedec(&signal, Wavelet::Haar, Some(2), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 46: `let reconstructed = waverec(&coeffs, Wavelet::Haar).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()