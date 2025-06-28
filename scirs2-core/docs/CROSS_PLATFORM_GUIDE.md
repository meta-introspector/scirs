# Cross-Platform Guide for scirs2-core

This guide provides information about using scirs2-core across different platforms and architectures.

## Supported Platforms

### Tier 1 (Fully Supported)
- **Linux x86_64**: Ubuntu 20.04+, Debian 10+, RHEL 8+, Fedora 32+
- **macOS x86_64**: macOS 10.15+ (Catalina and later)
- **macOS ARM64**: macOS 11.0+ (Big Sur and later on Apple Silicon)
- **Windows x86_64**: Windows 10/11, Windows Server 2019+

### Tier 2 (Best Effort)
- **Linux ARM64**: Ubuntu/Debian on ARM64
- **WebAssembly**: wasm32-unknown-unknown target
- **Linux ARM32**: armv7-unknown-linux-gnueabihf

## Platform-Specific Features

### Linux

#### Dependencies
```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev liblapack-dev

# Fedora/RHEL
sudo dnf install openblas-devel lapack-devel

# Arch Linux
sudo pacman -S openblas lapack
```

#### Recommended Features
```toml
[dependencies]
scirs2-core = { version = "0.1.0-beta.1", features = ["openblas", "memory_efficient", "simd", "parallel"] }
```

### macOS

#### Built-in Accelerate Framework
macOS includes the Accelerate framework, providing optimized BLAS/LAPACK implementations.

#### Recommended Features
```toml
[dependencies]
scirs2-core = { version = "0.1.0-beta.1", features = ["accelerate", "memory_efficient", "simd", "parallel"] }
```

#### Apple Silicon Considerations
- Native ARM64 builds are fully supported
- Rosetta 2 translation works but with performance penalty
- Use native ARM64 Rust toolchain for best performance

### Windows

#### Dependencies
- Visual Studio 2019 or later (for MSVC toolchain)
- Or MinGW-w64 for GNU toolchain

#### BLAS/LAPACK Options
1. Intel MKL (recommended for Intel CPUs)
2. OpenBLAS (prebuilt binaries available)
3. Reference BLAS/LAPACK

#### Recommended Features
```toml
[dependencies]
scirs2-core = { version = "0.1.0-beta.1", features = ["memory_efficient", "simd", "parallel"] }
```

### WebAssembly

#### Limitations
- No file system access (memory_efficient features limited)
- No threading (parallel features disabled)
- Limited SIMD support (depends on browser)

#### Build Command
```bash
# Add WASM target
rustup target add wasm32-unknown-unknown

# Build for WASM
cargo build --target wasm32-unknown-unknown --no-default-features --features "array,validation"
```

## Feature Compatibility Matrix

| Feature | Linux | macOS | Windows | WASM |
|---------|-------|-------|---------|------|
| array | ✅ | ✅ | ✅ | ✅ |
| validation | ✅ | ✅ | ✅ | ✅ |
| simd | ✅ | ✅ | ✅ | ⚠️ |
| parallel | ✅ | ✅ | ✅ | ❌ |
| memory_efficient | ✅ | ✅ | ✅ | ⚠️ |
| gpu | ✅ | ✅ | ✅ | ❌ |
| cuda | ✅ | ❌ | ✅ | ❌ |
| metal | ❌ | ✅ | ❌ | ❌ |
| opencl | ✅ | ✅ | ✅ | ❌ |

Legend: ✅ Full support, ⚠️ Partial support, ❌ Not supported

## Performance Considerations

### SIMD Instructions
- **x86_64**: SSE2 baseline, AVX/AVX2 when available
- **ARM64**: NEON instructions
- **WASM**: SIMD128 when supported by runtime

### Memory Mapping
- **Linux/macOS**: Full mmap support
- **Windows**: Memory mapping via Windows API
- **WASM**: Not available

### Threading
- **Desktop**: Uses all available CPU cores by default
- **WASM**: Single-threaded only

## Build Optimization

### Release Builds
```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
```

### Platform-Specific Optimizations
```toml
# For x86_64 with modern CPU
[profile.release]
target-cpu = "native"

# For specific ARM targets
[profile.release]
target-cpu = "cortex-a72"  # Example for Raspberry Pi 4
```

## Testing Across Platforms

### Local Testing
```bash
# Run the cross-platform validation script
./scripts/cross_platform_validation.sh

# Test specific platform features
cargo test --features "memory_efficient,simd,parallel"
```

### CI/CD Integration
Use the provided GitHub Actions workflow:
```yaml
# .github/workflows/cross_platform_ci.yml
# Automatically tests on Linux, macOS, Windows, and WASM
```

## Troubleshooting

### Common Linux Issues
- **Missing BLAS**: Install OpenBLAS or Intel MKL
- **Old glibc**: Update system or use static linking

### Common macOS Issues
- **Xcode Command Line Tools**: Install with `xcode-select --install`
- **Homebrew conflicts**: Ensure consistent library versions

### Common Windows Issues
- **Long path names**: Enable long path support in Windows
- **Missing MSVC**: Install Visual Studio Build Tools

### WASM Issues
- **Module size**: Use `--features` to minimize binary size
- **Browser compatibility**: Test in multiple browsers

## Platform-Specific Examples

### Linux with OpenBLAS
```rust
use scirs2_core::prelude::*;

#[cfg(all(target_os = "linux", feature = "openblas"))]
fn optimized_computation() {
    // Linux-specific optimized code
}
```

### macOS with Metal GPU
```rust
#[cfg(all(target_os = "macos", feature = "metal"))]
fn gpu_computation() {
    // macOS Metal-specific code
}
```

### Cross-Platform Fallback
```rust
#[cfg(feature = "simd")]
fn compute_simd() { /* SIMD implementation */ }

#[cfg(not(feature = "simd"))]
fn compute_simd() { /* Scalar fallback */ }
```

## Best Practices

1. **Always test on target platforms**: Don't assume cross-compilation catches all issues
2. **Use conditional compilation**: Provide platform-specific optimizations
3. **Document platform requirements**: Be clear about dependencies
4. **Provide fallbacks**: Ensure functionality without platform-specific features
5. **Monitor CI**: Regularly check cross-platform CI results

## Getting Help

- **Issue Tracker**: Report platform-specific bugs
- **Documentation**: Check platform-specific notes
- **Community**: Discord channel for platform support

---

*Last Updated: 2025-06-28*