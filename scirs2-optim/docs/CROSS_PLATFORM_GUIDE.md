# Cross-Platform Compatibility Guide

This guide provides comprehensive information for building, testing, and deploying scirs2-optim across different platforms and architectures.

## Table of Contents

1. [Supported Platforms](#supported-platforms)
2. [Platform-Specific Setup](#platform-specific-setup)
3. [Cross-Compilation](#cross-compilation)
4. [Platform-Specific Optimizations](#platform-specific-optimizations)
5. [Testing Strategy](#testing-strategy)
6. [Troubleshooting](#troubleshooting)
7. [Performance Considerations](#performance-considerations)

## Supported Platforms

### Primary Platforms (Tier 1)

| Platform | Architecture | Status | Notes |
|----------|--------------|--------|-------|
| Linux | x86_64 | ✅ Full Support | Primary development platform |
| Windows | x86_64 | ✅ Full Support | MSVC toolchain |
| macOS | x86_64 | ✅ Full Support | Intel Macs |
| macOS | aarch64 (Apple Silicon) | ✅ Full Support | M1/M2/M3 Macs |

### Secondary Platforms (Tier 2)

| Platform | Architecture | Status | Notes |
|----------|--------------|--------|-------|
| Linux | aarch64 | ✅ Cross-compilation | ARM64 Linux servers |
| Windows | aarch64 | ✅ Cross-compilation | ARM64 Windows |
| FreeBSD | x86_64 | ⚠️ Community Support | Limited testing |

### Experimental Platforms

| Platform | Architecture | Status | Notes |
|----------|--------------|--------|-------|
| WebAssembly | wasm32 | 🧪 Experimental | Limited functionality |
| Android | aarch64 | 🧪 Experimental | Via Termux |

## Platform-Specific Setup

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    pkg-config \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    gfortran \
    libssl-dev \
    curl

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Build scirs2-optim
git clone <repository>
cd scirs2-optim
cargo build --release --all-features
```

#### Red Hat/CentOS/Fedora

```bash
# Install dependencies
sudo dnf install -y \
    gcc \
    gcc-gfortran \
    pkg-config \
    openssl-devel \
    openblas-devel \
    lapack-devel

# Or for older versions
sudo yum install -y \
    gcc \
    gcc-gfortran \
    pkg-config \
    openssl-devel \
    openblas-devel \
    lapack-devel
```

#### Arch Linux

```bash
# Install dependencies
sudo pacman -S \
    base-devel \
    pkg-config \
    openssl \
    openblas \
    lapack \
    gfortran
```

### Windows

#### Prerequisites

1. **Visual Studio Build Tools**: Install Visual Studio 2019 or later with C++ build tools
2. **Rust**: Download from [rustup.rs](https://rustup.rs/)

```powershell
# Install via chocolatey (optional but recommended)
choco install rust
choco install llvm  # For additional toolchain support

# Or via winget
winget install Rustlang.Rustup
```

#### Building with MSVC

```powershell
# Clone and build
git clone <repository>
cd scirs2-optim
cargo build --release --all-features
```

#### Building with MinGW

```powershell
# Install MinGW target
rustup target add x86_64-pc-windows-gnu

# Install MinGW-w64
# Via MSYS2 or chocolatey
choco install mingw

# Build with GNU toolchain
cargo build --target x86_64-pc-windows-gnu --release
```

#### External Dependencies

For BLAS/LAPACK support on Windows:

```powershell
# Using vcpkg (recommended)
git clone https://github.com/Microsoft/vcpkg.git C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
C:\vcpkg\vcpkg install openblas:x64-windows

# Set environment variables
$env:VCPKG_ROOT = "C:\vcpkg"
$env:CMAKE_TOOLCHAIN_FILE = "C:\vcpkg\scripts\buildsystems\vcpkg.cmake"
```

### macOS

#### Prerequisites

```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

#### Dependencies

```bash
# Install mathematical libraries
brew install openblas lapack pkg-config openssl

# Set environment variables for building
export PKG_CONFIG_PATH="$(brew --prefix)/lib/pkgconfig"
export OPENSSL_DIR="$(brew --prefix openssl)"
export OPENBLAS_DIR="$(brew --prefix openblas)"
```

#### Apple Silicon (M1/M2/M3) Specific

```bash
# For native ARM64 builds
rustup target add aarch64-apple-darwin

# Build native ARM64
cargo build --target aarch64-apple-darwin --release

# For universal binaries (Intel + ARM64)
rustup target add x86_64-apple-darwin
cargo build --target x86_64-apple-darwin --release
cargo build --target aarch64-apple-darwin --release

# Combine into universal binary (optional)
lipo -create \
    target/x86_64-apple-darwin/release/scirs2-optim \
    target/aarch64-apple-darwin/release/scirs2-optim \
    -output target/universal/scirs2-optim
```

## Cross-Compilation

### Setting Up Cross-Compilation

```bash
# Install cross-compilation tool
cargo install cross

# Add targets
rustup target add aarch64-unknown-linux-gnu
rustup target add x86_64-pc-windows-gnu
rustup target add aarch64-apple-darwin
```

### Linux to Windows

```bash
# Install MinGW cross-compiler
sudo apt-get install gcc-mingw-w64

# Cross-compile
cargo build --target x86_64-pc-windows-gnu --release
```

### Linux to ARM64

```bash
# Install ARM64 cross-compiler
sudo apt-get install gcc-aarch64-linux-gnu

# Cross-compile using cross
cross build --target aarch64-unknown-linux-gnu --release
```

### Cross-Compilation Configuration

Create `.cargo/config.toml`:

```toml
[target.aarch64-unknown-linux-gnu]
linker = "aarch64-linux-gnu-gcc"

[target.x86_64-pc-windows-gnu]
linker = "x86_64-w64-mingw32-gcc"

[target.aarch64-pc-windows-msvc]
# Use cross for this target
runner = "cross"

[build]
# Default to native optimization
rustflags = ["-C", "target-cpu=native"]

[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-cpu=native", "-C", "link-arg=-fuse-ld=lld"]

[target.aarch64-apple-darwin]
rustflags = ["-C", "target-cpu=apple-m1"]
```

## Platform-Specific Optimizations

### SIMD Optimizations

#### Intel/AMD x86_64

```rust
// Enable AVX2 for better performance on modern CPUs
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_optimized_operation(data: &[f64]) -> f64 {
    // AVX2-optimized implementation
}
```

#### ARM64 (NEON)

```rust
// ARM NEON optimizations
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_optimized_operation(data: &[f64]) -> f64 {
    // NEON-optimized implementation
}
```

### Platform-Specific Features

```toml
# Cargo.toml feature flags
[features]
default = []

# Platform-specific SIMD
simd_avx2 = []      # x86_64 AVX2
simd_neon = []      # ARM64 NEON
simd_wasm = []      # WebAssembly SIMD

# Platform-specific acceleration
accelerate = []     # macOS Accelerate framework
mkl = []           # Intel MKL
openblas = []      # OpenBLAS
cuda = []          # NVIDIA CUDA
rocm = []          # AMD ROCm

[target.'cfg(target_os = "macos")'.dependencies]
accelerate-src = { version = "0.3", optional = true }

[target.'cfg(target_os = "linux")'.dependencies]
openblas-src = { version = "0.10", optional = true }

[target.'cfg(target_os = "windows")'.dependencies]
intel-mkl-src = { version = "0.8", optional = true }
```

### Performance Tuning by Platform

#### Linux Performance Tuning

```bash
# Set CPU governor for performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable CPU frequency scaling during benchmarks
sudo cpupower frequency-set --governor performance

# Set process priority
nice -n -20 cargo run --release --example benchmark

# Use specific NUMA node if available
numactl --cpunodebind=0 --membind=0 cargo run --release
```

#### Windows Performance Tuning

```powershell
# Set high performance power plan
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# Increase process priority
$process = Get-Process -Name "scirs2-optim"
$process.PriorityClass = "High"
```

#### macOS Performance Tuning

```bash
# Disable App Nap for better performance
defaults write NSGlobalDomain NSAppSleepDisabled -bool YES

# For benchmarking, disable thermal throttling monitoring
sudo pmset -a disablesleep 1
```

## Testing Strategy

### Automated Cross-Platform Testing

Our CI/CD pipeline automatically tests on multiple platforms:

```yaml
# GitHub Actions matrix testing
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    rust: [stable, beta]
    features: [default, simd, parallel, all-features]
```

### Platform-Specific Test Suites

#### Linux-Specific Tests

```bash
# Memory testing with Valgrind
cargo build --features debug-assertions
valgrind --tool=memcheck --leak-check=full \
    ./target/debug/examples/optimizer_test

# Performance profiling with perf
perf record -g cargo run --release --example benchmark
perf report

# NUMA awareness testing
numactl --hardware
for node in $(numactl --hardware | grep "available:" | cut -d' ' -f2); do
    numactl --cpunodebind=$node --membind=$node \
        cargo test --release numa_test
done
```

#### Windows-Specific Tests

```powershell
# Test different Windows versions
# Run on Windows Server, Windows 10, Windows 11

# Test with different Visual Studio versions
# VS2019, VS2022

# Performance counters
Get-Counter "\Process(scirs2-optim)\% Processor Time"
Get-Counter "\Process(scirs2-optim)\Working Set"
```

#### macOS-Specific Tests

```bash
# Test on both Intel and Apple Silicon
arch -x86_64 cargo test --release
arch -arm64 cargo test --release

# Test with Instruments
instruments -t "Time Profiler" cargo run --release --example benchmark

# Test memory with leaks
leaks --atExit -- ./target/release/examples/optimizer_test
```

### Feature Compatibility Matrix

| Feature | Linux x86_64 | Linux ARM64 | Windows x86_64 | Windows ARM64 | macOS Intel | macOS ARM64 |
|---------|--------------|-------------|----------------|---------------|-------------|-------------|
| SIMD | ✅ AVX2 | ✅ NEON | ✅ AVX2 | ✅ NEON | ✅ AVX2 | ✅ NEON |
| Parallel | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| GPU (CUDA) | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| GPU (ROCm) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| GPU (Metal) | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| OpenBLAS | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Accelerate | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Intel MKL | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ |

## Troubleshooting

### Common Build Issues

#### Linux Issues

**Problem**: Missing BLAS/LAPACK libraries
```bash
# Solution: Install development packages
sudo apt-get install libblas-dev liblapack-dev libopenblas-dev
```

**Problem**: Linker errors with OpenBLAS
```bash
# Solution: Set PKG_CONFIG_PATH
export PKG_CONFIG_PATH="/usr/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH"
```

#### Windows Issues

**Problem**: MSVC not found
```powershell
# Solution: Install Visual Studio Build Tools
# Or use rustup default stable-x86_64-pc-windows-gnu
```

**Problem**: OpenBLAS linking errors
```powershell
# Solution: Use vcpkg or pre-built binaries
vcpkg install openblas:x64-windows
```

#### macOS Issues

**Problem**: OpenSSL not found
```bash
# Solution: Set OPENSSL_DIR
export OPENSSL_DIR=$(brew --prefix openssl)
```

**Problem**: Apple Silicon compilation errors
```bash
# Solution: Update Xcode and use correct target
rustup target add aarch64-apple-darwin
export MACOSX_DEPLOYMENT_TARGET=11.0
```

### Performance Issues

#### Cross-Platform Performance Debugging

```bash
# 1. Check CPU features
rustc --print cfg | grep target_feature

# 2. Verify SIMD usage
cargo asm --release --example benchmark | grep -E "(vmov|vadd|vfma)"

# 3. Profile memory allocation
cargo install dhat
DHAT_PROFILING=1 cargo run --example memory_benchmark

# 4. Check for platform-specific bottlenecks
cargo install flamegraph
cargo flamegraph --example benchmark
```

### Platform-Specific Debugging

#### Linux Debugging

```bash
# GDB debugging
cargo build --features debug-assertions
gdb ./target/debug/examples/optimizer_test

# Strace for system call analysis
strace -c cargo run --release --example benchmark

# Memory mapping analysis
pmap $(pgrep scirs2-optim)
```

#### Windows Debugging

```powershell
# Visual Studio debugger
# Build with debug info: cargo build --release --features debug-assertions

# Performance Toolkit
# Download Windows Performance Toolkit
# Use Windows Performance Analyzer (WPA)

# Process Monitor
# Download ProcMon from Microsoft Sysinternals
```

#### macOS Debugging

```bash
# Xcode debugger
lldb ./target/debug/examples/optimizer_test

# Instruments profiling
instruments -t "Allocations" cargo run --release --example benchmark
instruments -t "Time Profiler" cargo run --release --example benchmark

# System call tracing
dtruss -n cargo run --release --example benchmark
```

## Performance Considerations

### Platform-Specific Optimizations

#### Compilation Flags

```bash
# Linux - maximum performance
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C codegen-units=1 -C lto=fat" \
    cargo build --release

# Windows - compatible performance
RUSTFLAGS="-C target-cpu=skylake -C opt-level=3" \
    cargo build --release

# macOS Intel - maximum performance
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C codegen-units=1" \
    cargo build --release

# macOS Apple Silicon - optimized for M1/M2
RUSTFLAGS="-C target-cpu=apple-m1 -C opt-level=3" \
    cargo build --release
```

#### Runtime Optimization

```rust
// Platform-specific runtime detection
use std::sync::Once;

static INIT: Once = Once::new();
static mut OPTIMIZATION_LEVEL: OptimizationLevel = OptimizationLevel::Default;

#[derive(Clone, Copy)]
enum OptimizationLevel {
    Default,
    SIMD,
    AVX2,
    NEON,
}

fn detect_platform_optimizations() -> OptimizationLevel {
    INIT.call_once(|| {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    OPTIMIZATION_LEVEL = OptimizationLevel::AVX2;
                } else if is_x86_feature_detected!("sse4.1") {
                    OPTIMIZATION_LEVEL = OptimizationLevel::SIMD;
                }
            }
            
            #[cfg(target_arch = "aarch64")]
            {
                if std::arch::is_aarch64_feature_detected!("neon") {
                    OPTIMIZATION_LEVEL = OptimizationLevel::NEON;
                }
            }
        }
    });
    
    unsafe { OPTIMIZATION_LEVEL }
}
```

### Benchmarking Across Platforms

```rust
// Cross-platform benchmarking harness
use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_optimizer_cross_platform(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_platform");
    
    // Configure for platform-specific timing
    #[cfg(target_os = "linux")]
    group.measurement_time(std::time::Duration::from_secs(10));
    
    #[cfg(target_os = "windows")]
    group.measurement_time(std::time::Duration::from_secs(15)); // Account for Windows overhead
    
    #[cfg(target_os = "macos")]
    group.measurement_time(std::time::Duration::from_secs(8));  // Generally faster I/O
    
    group.bench_function("sgd_step", |b| {
        b.iter(|| {
            // Platform-optimized benchmark
        });
    });
    
    group.finish();
}

criterion_group!(benches, benchmark_optimizer_cross_platform);
criterion_main!(benches);
```

## Distribution and Deployment

### Binary Distribution

```bash
# Build release binaries for all platforms
cargo install cross

# Linux
cross build --target x86_64-unknown-linux-gnu --release
cross build --target aarch64-unknown-linux-gnu --release

# Windows
cross build --target x86_64-pc-windows-gnu --release
cargo build --target x86_64-pc-windows-msvc --release

# macOS
cargo build --target x86_64-apple-darwin --release
cargo build --target aarch64-apple-darwin --release
```

### Platform-Specific Packages

#### Linux (AppImage/Flatpak)

```bash
# Create AppImage
# Use linuxdeploy or appimage-builder

# Create Flatpak
# Define org.scirs.scirs2-optim.yaml manifest
```

#### Windows (MSI/NSIS)

```powershell
# Use WiX Toolset for MSI
# Or NSIS for installer
```

#### macOS (DMG/PKG)

```bash
# Create .app bundle
# Use create-dmg for DMG creation
# Use pkgbuild for PKG creation
```

### Container Distribution

```dockerfile
# Multi-platform Dockerfile
FROM --platform=$TARGETPLATFORM rust:1.70 AS builder

ARG TARGETPLATFORM
ARG BUILDPLATFORM

WORKDIR /usr/src/app
COPY . .

RUN case "$TARGETPLATFORM" in \
    "linux/amd64") \
        cargo build --release --target x86_64-unknown-linux-gnu ;; \
    "linux/arm64") \
        cargo build --release --target aarch64-unknown-linux-gnu ;; \
    esac

FROM --platform=$TARGETPLATFORM debian:bullseye-slim
RUN apt-get update && apt-get install -y libopenblas0 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/src/app/target/*/release/scirs2-optim /usr/local/bin/
ENTRYPOINT ["scirs2-optim"]
```

## Continuous Integration

Our cross-platform CI ensures compatibility across all supported platforms:

- **GitHub Actions**: Primary CI for Linux, Windows, macOS
- **Cross-compilation testing**: ARM64 and other architectures
- **Performance regression testing**: Platform-specific benchmarks
- **Feature matrix testing**: All feature combinations on all platforms

See `.github/workflows/cross_platform_tests.yml` for the complete testing strategy.

## Contributing

When contributing cross-platform features:

1. Test on at least 2 different platforms
2. Use platform-specific conditional compilation appropriately
3. Document any platform-specific requirements
4. Update the compatibility matrix if needed
5. Ensure cross-compilation works for your changes

## Support

For platform-specific issues:

- **Linux**: Check distribution-specific package repositories
- **Windows**: Verify Visual Studio Build Tools installation
- **macOS**: Ensure Xcode Command Line Tools are current
- **Cross-compilation**: Use the `cross` tool for consistent results

For questions and support, please open an issue with your platform details and error messages.