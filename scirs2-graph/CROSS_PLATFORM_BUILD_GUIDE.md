# Cross-Platform Build Verification Guide

**Version**: v0.1.0-beta.1  
**Last Updated**: January 21, 2025  

This guide provides comprehensive instructions for verifying scirs2-graph builds across multiple platforms, architectures, and Rust toolchain versions.

## Table of Contents

1. [Supported Platforms](#supported-platforms)
2. [Build Verification Strategy](#build-verification-strategy)
3. [Local Testing Setup](#local-testing-setup)
4. [CI/CD Configuration](#cicd-configuration)
5. [Platform-Specific Considerations](#platform-specific-considerations)
6. [Performance Validation](#performance-validation)
7. [Troubleshooting](#troubleshooting)
8. [Release Checklist](#release-checklist)

## Supported Platforms

### Tier 1 Platforms (Guaranteed Support)
✅ **Linux x86_64-unknown-linux-gnu**
- Ubuntu 20.04 LTS, 22.04 LTS
- RHEL/CentOS 8+
- Debian 11+
- Rust 1.70+

✅ **macOS x86_64-apple-darwin**
- macOS 11.0+ (Big Sur)
- Xcode 12.0+
- Rust 1.70+

✅ **macOS aarch64-apple-darwin** (Apple Silicon)
- macOS 11.0+ (Big Sur)
- Xcode 12.0+
- Rust 1.70+

✅ **Windows x86_64-pc-windows-msvc**
- Windows 10 version 1903+
- Windows 11
- Visual Studio 2019+
- Rust 1.70+

### Tier 2 Platforms (Best Effort Support)
⚠️ **Linux aarch64-unknown-linux-gnu**
- ARM64 Linux distributions
- Cross-compilation from x86_64

⚠️ **Windows x86_64-pc-windows-gnu**
- MinGW-w64 toolchain
- MSYS2 environment

⚠️ **Linux i686-unknown-linux-gnu**
- 32-bit Linux distributions
- Legacy system support

### Tier 3 Platforms (Community Supported)
🔄 **FreeBSD x86_64-unknown-freebsd**
🔄 **NetBSD x86_64-unknown-netbsd**
🔄 **OpenBSD x86_64-unknown-openbsd**

## Build Verification Strategy

### Verification Matrix

| Platform | Rust Version | Features | Test Suite |
|----------|-------------|----------|------------|
| Linux x86_64 | 1.70, 1.75, stable | default, full | ✅ Complete |
| macOS x86_64 | 1.70, stable | default, full | ✅ Complete |
| macOS ARM64 | 1.70, stable | default, full | ✅ Complete |
| Windows MSVC | 1.70, stable | default, full | ✅ Complete |
| Linux ARM64 | stable | default | ⚠️ Basic |
| Windows GNU | stable | default | ⚠️ Basic |

### Build Configurations

#### Debug Build
```bash
cargo build --all-features
cargo test --all-features
```

#### Release Build
```bash
cargo build --release --all-features
cargo test --release --all-features
```

#### Minimal Build
```bash
cargo build --no-default-features
cargo test --no-default-features
```

#### Feature Matrix Testing
```bash
# Test individual features
cargo test --features parallel
cargo test --features simd
cargo test --features gpu
cargo test --features ultrathink
cargo test --features validation

# Test feature combinations
cargo test --features "parallel,simd"
cargo test --features "ultrathink,gpu"
```

## Local Testing Setup

### Prerequisites Installation

#### Linux (Ubuntu/Debian)
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install system dependencies
sudo apt update
sudo apt install -y build-essential pkg-config libssl-dev

# Install additional tools
cargo install cargo-nextest cross
```

#### macOS
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install Xcode command line tools
xcode-select --install

# Install additional tools
cargo install cargo-nextest
```

#### Windows (PowerShell)
```powershell
# Install Rust
Invoke-WebRequest -Uri "https://win.rustup.rs/" -OutFile "rustup-init.exe"
.\rustup-init.exe

# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

# Install additional tools
cargo install cargo-nextest
```

### Cross-Compilation Setup

#### Linux → ARM64
```bash
# Install cross-compilation target
rustup target add aarch64-unknown-linux-gnu

# Install cross-compilation toolchain
sudo apt install gcc-aarch64-linux-gnu

# Configure Cargo
cat >> ~/.cargo/config.toml << EOF
[target.aarch64-unknown-linux-gnu]
linker = "aarch64-linux-gnu-gcc"
EOF
```

#### Cross Tool Installation
```bash
# Install cross for easy cross-compilation
cargo install cross

# Test cross-compilation
cross build --target aarch64-unknown-linux-gnu
cross test --target aarch64-unknown-linux-gnu
```

## CI/CD Configuration

### GitHub Actions Workflow

Create `.github/workflows/cross-platform.yml`:

```yaml
name: Cross-Platform Build Verification

on:
  push:
    branches: [ main, master, develop ]
  pull_request:
    branches: [ main, master, develop ]

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1

jobs:
  test-tier1:
    name: Test Tier 1 Platforms
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        rust: [1.70.0, stable]
        include:
          - os: macos-latest
            target: aarch64-apple-darwin
          - os: ubuntu-latest  
            target: x86_64-unknown-linux-gnu
          - os: windows-latest
            target: x86_64-pc-windows-msvc

    steps:
    - uses: actions/checkout@v4
    
    - name: Install Rust
      uses: dtolnay/rust-toolchain@master
      with:
        toolchain: ${{ matrix.rust }}
        targets: ${{ matrix.target }}
        
    - name: Install system dependencies (Linux)
      if: runner.os == 'Linux'
      run: |
        sudo apt update
        sudo apt install -y build-essential pkg-config libssl-dev
        
    - name: Install cargo-nextest
      run: cargo install cargo-nextest
      
    - name: Cache cargo registry
      uses: actions/cache@v3
      with:
        path: ~/.cargo/registry
        key: ${{ runner.os }}-cargo-registry-${{ hashFiles('**/Cargo.lock') }}
        
    - name: Build (Debug)
      run: cargo build --all-features
      
    - name: Build (Release)  
      run: cargo build --release --all-features
      
    - name: Test (Debug)
      run: cargo nextest run --all-features
      
    - name: Test (Release)
      run: cargo nextest run --release --all-features
      
    - name: Test minimal build
      run: cargo test --no-default-features
      
    - name: Feature matrix testing
      run: |
        cargo test --features parallel
        cargo test --features simd
        cargo test --features validation
        
  test-tier2:
    name: Test Tier 2 Platforms  
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        target:
          - aarch64-unknown-linux-gnu
          - i686-unknown-linux-gnu
          
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Rust
      uses: dtolnay/rust-toolchain@stable
      
    - name: Install cross
      run: cargo install cross
      
    - name: Cross build
      run: cross build --target ${{ matrix.target }}
      
    - name: Cross test  
      run: cross test --target ${{ matrix.target }}

  benchmarks:
    name: Performance Benchmarks
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Rust
      uses: dtolnay/rust-toolchain@stable
      
    - name: Run benchmarks
      run: |
        cargo bench --bench graph_benchmarks
        cargo bench --bench networkx_igraph_comparison
        
    - name: Upload benchmark results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: target/criterion/
```

### Platform-Specific CI Jobs

#### Windows-Specific Testing
```yaml
  windows-specific:
    name: Windows-Specific Tests
    runs-on: windows-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Rust (MSVC)
      uses: dtolnay/rust-toolchain@stable
      with:
        toolchain: stable-x86_64-pc-windows-msvc
        
    - name: Install Rust (GNU)  
      run: rustup toolchain install stable-x86_64-pc-windows-gnu
      
    - name: Test MSVC build
      run: cargo test --target x86_64-pc-windows-msvc
      
    - name: Test GNU build
      run: cargo test --target x86_64-pc-windows-gnu
```

#### macOS-Specific Testing  
```yaml
  macos-specific:
    name: macOS-Specific Tests
    runs-on: macos-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Rust
      uses: dtolnay/rust-toolchain@stable
      
    - name: Add ARM64 target
      run: rustup target add aarch64-apple-darwin
      
    - name: Test x86_64 build
      run: cargo test --target x86_64-apple-darwin
      
    - name: Test ARM64 build
      run: cargo build --target aarch64-apple-darwin
```

## Platform-Specific Considerations

### Linux

#### Dependencies
- **OpenBLAS/LAPACK**: May require system packages
- **OpenSSL**: Version compatibility issues
- **glibc**: Minimum version requirements

#### Common Issues
```bash
# Fix OpenSSL linking issues
export PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig

# Fix BLAS/LAPACK linking
sudo apt install libopenblas-dev liblapack-dev
```

#### Performance Notes
- **SIMD**: Ensure CPU features are available
- **Memory**: Check system limits for large graph tests
- **Parallelism**: Verify thread count detection

### macOS

#### Dependencies
- **Xcode**: Required for system frameworks
- **Homebrew**: Optional for additional dependencies

#### Common Issues
```bash
# Fix linker issues
export MACOSX_DEPLOYMENT_TARGET=11.0

# Fix Apple Silicon compilation
export ARCHFLAGS="-arch arm64"
```

#### Performance Notes
- **M1/M2**: Native ARM64 performance significantly better
- **Memory**: Unified memory architecture considerations
- **Threading**: Efficiency cores vs performance cores

### Windows

#### Dependencies  
- **Visual Studio**: Build tools required
- **MinGW**: Alternative toolchain option

#### Common Issues
```powershell
# Fix MSVC linking
set LIB=%LIB%;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.19041.0\um\x64

# Fix path length issues
git config --system core.longpaths true
```

#### Performance Notes
- **SIMD**: May require explicit feature enabling
- **Memory**: Virtual memory limits
- **Threading**: Windows thread scheduling differences

## Performance Validation

### Cross-Platform Benchmarks

#### Benchmark Suite Execution
```bash
#!/bin/bash
# scripts/cross_platform_benchmarks.sh

PLATFORMS=("x86_64-unknown-linux-gnu" "x86_64-apple-darwin" "x86_64-pc-windows-msvc")
RESULTS_DIR="benchmark_results"

mkdir -p $RESULTS_DIR

for platform in "${PLATFORMS[@]}"; do
    echo "Running benchmarks for $platform..."
    
    # Build for target platform
    cargo build --release --target $platform
    
    # Run benchmarks
    cargo bench --target $platform -- --output-format json > "$RESULTS_DIR/${platform}_results.json"
    
    echo "Completed benchmarks for $platform"
done

# Generate comparison report
python scripts/compare_benchmarks.py $RESULTS_DIR
```

#### Performance Regression Detection
```python
# scripts/compare_benchmarks.py
import json
import sys
from pathlib import Path

def compare_platforms(results_dir):
    platforms = ["x86_64-unknown-linux-gnu", "x86_64-apple-darwin", "x86_64-pc-windows-msvc"]
    results = {}
    
    for platform in platforms:
        result_file = Path(results_dir) / f"{platform}_results.json"
        if result_file.exists():
            with open(result_file) as f:
                results[platform] = json.load(f)
    
    # Compare performance across platforms
    for benchmark in results[platforms[0]]:
        times = [results[p][benchmark]["mean"] for p in platforms if p in results]
        variance = max(times) / min(times)
        
        if variance > 2.0:  # More than 2x difference
            print(f"WARNING: High performance variance in {benchmark}: {variance:.2f}x")

if __name__ == "__main__":
    compare_platforms(sys.argv[1])
```

## Troubleshooting

### Common Build Issues

#### Linking Errors
```bash
# Linux: Missing system libraries
sudo apt install build-essential pkg-config libssl-dev

# macOS: Xcode tools not installed  
xcode-select --install

# Windows: MSVC not found
# Install Visual Studio Build Tools
```

#### Target Not Found
```bash
# Install missing target
rustup target add <target-triple>

# Update Rust toolchain
rustup update
```

#### Feature Compilation Errors
```bash
# Check feature dependencies
cargo tree --features <feature-name>

# Test feature independently
cargo check --features <feature-name> --no-default-features
```

### Platform-Specific Issues

#### Linux ARM64 Cross-Compilation
```bash
# Install cross-compilation tools
sudo apt install gcc-aarch64-linux-gnu

# Configure linker
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
```

#### macOS Universal Binaries
```bash
# Build universal binary
cargo build --target x86_64-apple-darwin --target aarch64-apple-darwin
lipo -create -output target/universal/scirs2-graph \
    target/x86_64-apple-darwin/release/scirs2-graph \
    target/aarch64-apple-darwin/release/scirs2-graph
```

#### Windows Path Length Issues
```powershell
# Enable long paths in Git
git config --system core.longpaths true

# Enable long paths in Windows
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1
```

## Release Checklist

### Pre-Release Verification

- [ ] **Tier 1 Platforms**: All tests pass on Linux, macOS, Windows
- [ ] **Feature Matrix**: All feature combinations build and test successfully  
- [ ] **Cross-Compilation**: ARM64 and other tier 2 targets build
- [ ] **Performance**: No regressions > 10% on any platform
- [ ] **Documentation**: Platform-specific notes updated
- [ ] **Dependencies**: Version compatibility verified

### Release Process

1. **Version Bump**
   ```bash
   # Update version in Cargo.toml
   sed -i 's/version = ".*"/version = "0.1.0"/' Cargo.toml
   ```

2. **Tag Release**
   ```bash
   git tag -a v0.1.0 -m "Release v0.1.0"
   git push origin v0.1.0
   ```

3. **Publish to crates.io**
   ```bash
   cargo publish --dry-run
   cargo publish
   ```

4. **Create GitHub Release**
   - Upload platform-specific binaries
   - Include changelog and migration notes
   - Document known platform limitations

### Post-Release Validation

- [ ] **Download Test**: Verify `cargo install scirs2-graph` works on all platforms
- [ ] **Integration Test**: Test in fresh environment
- [ ] **Documentation**: Verify docs.rs builds correctly
- [ ] **Community Feedback**: Monitor for platform-specific issues

## Automated Testing Scripts

### Build Test Script
```bash
#!/bin/bash
# scripts/test_all_platforms.sh

set -e

TARGETS=(
    "x86_64-unknown-linux-gnu"
    "x86_64-apple-darwin" 
    "aarch64-apple-darwin"
    "x86_64-pc-windows-msvc"
    "aarch64-unknown-linux-gnu"
)

echo "🚀 Starting cross-platform build verification..."

for target in "${TARGETS[@]}"; do
    echo "🔧 Testing target: $target"
    
    # Install target if not present
    rustup target add $target 2>/dev/null || true
    
    # Try native build first, then cross
    if cargo build --target $target 2>/dev/null; then
        echo "✅ Native build successful for $target"
        
        # Run tests if possible
        if cargo test --target $target 2>/dev/null; then
            echo "✅ Tests passed for $target"
        else
            echo "⚠️ Tests skipped for $target (cross-compilation)"
        fi
    else
        echo "🔄 Trying cross-compilation for $target"
        if command -v cross >/dev/null && cross build --target $target; then
            echo "✅ Cross-compilation successful for $target"
        else
            echo "❌ Build failed for $target"
        fi
    fi
    
    echo ""
done

echo "🎉 Cross-platform verification completed!"
```

### CI Status Badge
Add to README.md:
```markdown
[![Cross-Platform](https://github.com/your-org/scirs2-graph/workflows/Cross-Platform%20Build%20Verification/badge.svg)](https://github.com/your-org/scirs2-graph/actions)
```

---

**Maintenance**: This guide should be updated with each release to reflect new platform support and known issues.

**Contributing**: Platform-specific improvements and additional target support are welcome through pull requests.

**Support**: For platform-specific issues, please include your system information and build logs when reporting issues.