# SciRS2 Build Optimization Guide

## Current Status
- **Total modules**: 26 
- **Target directory size**: ~15GB
- **Duplicate dependencies**: 898+ instances
- **Current dev build time**: ~1s for check, >2min for full release build

## Optimization Strategies Implemented

### 1. Cargo Configuration (`/.cargo/config.toml`)
- **Faster linker**: Using LLD linker with `-fuse-ld=lld`
- **CPU optimization**: `-target-cpu=native` for current hardware
- **Parallel builds**: 8 jobs configured
- **Reduced debug info**: `debug = 1` for dev builds
- **Incremental compilation**: Enabled for development

### 2. Dependency Optimization (`Cargo.toml`)
- **Default features disabled**: For heavy dependencies like `ndarray-linalg`, `openblas-src`
- **Feature consolidation**: Organized dependencies by category
- **Workspace inheritance**: Consistent versions across modules

### 3. Profile Optimization
```toml
[profile.dev]
debug = 1           # Reduced debug info
opt-level = 0       # No optimization for fastest compilation
incremental = true

[profile.release]  
opt-level = 3
lto = "thin"        # Balanced LTO
codegen-units = 1
strip = true        # Smaller binaries
```

## Further Optimization Recommendations

### 4. Feature Gates for Heavy Dependencies
```toml
# In individual module Cargo.toml files
[features]
default = ["std"]
std = []
openblas = ["ndarray-linalg/openblas"]
simd = ["wide"]
parallel = ["rayon"]
```

### 5. Compilation Unit Reduction
- **Split large modules**: Break files >1000 lines into smaller units
- **Lazy feature loading**: Use feature gates for optional functionality
- **Conditional compilation**: `#[cfg(feature = "...")]` for expensive code

### 6. Dependency Deduplication
Current duplicates to consolidate:
- `approx` (0.3.2 vs 0.5.1)
- `bitflags` (1.3.2 vs 2.9.1) 
- `getrandom` (0.2.16 vs 0.3.3)
- `half` (1.8.3 vs 2.6.0)

### 7. Build Cache Optimization
```bash
# Use sccache for distributed compilation
export RUSTC_WRAPPER=sccache

# Clean target selectively
cargo clean -p specific-crate

# Use workspace members selectively
cargo build -p scirs2-core
```

### 8. Development Workflow
```bash
# Fast development cycle
cargo check              # Fastest syntax checking
cargo test --no-run      # Compile tests without running
cargo build --workspace  # Full workspace build
```

## Performance Targets
- **Dev builds**: <30s for full workspace
- **Release builds**: <5min for full workspace  
- **Incremental builds**: <5s for single module changes
- **Target size**: <10GB

## Implementation Priority
1. ✅ Cargo configuration optimization
2. ✅ Dependency feature reduction
3. 🔄 Module feature gating (in progress)
4. ⏳ Dependency deduplication
5. ⏳ Large module splitting
6. ⏳ Build cache setup

## Monitoring Build Performance
```bash
# Measure build times
time cargo build --workspace

# Analyze compilation units
cargo build --timings

# Check dependency tree
cargo tree --duplicates

# Monitor target directory growth
du -sh target/
```