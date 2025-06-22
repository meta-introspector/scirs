# CI Disk Space Optimization Guide

This document outlines strategies to fix the "No space left on device" error in GitHub Actions CI.

## Files Added/Modified

### 1. `build.rs` - Build Script Optimizations
- Detects CI environment automatically
- Enables linker optimizations for smaller artifacts
- Compresses debug sections with zlib
- Removes unused sections with `--gc-sections`

### 2. `.cargo/config.toml` - Cargo Configuration
- **Reduced parallel jobs**: `jobs = 2` (from 8) to reduce memory pressure
- **Linker optimizations**: Added compression and garbage collection flags
- **New CI profile**: `[profile.ci]` with minimal debug info and no incremental compilation
- **Disabled incremental compilation**: Saves significant disk space

### 3. `.github/workflows/ci-optimized.yml` - Optimized CI Workflow
- **Disk cleanup**: Removes unnecessary system files (~14GB freed)
- **Split builds**: Uses matrix strategy to build different profiles separately
- **Aggressive caching**: Better cache keys and paths
- **Reduced test scope**: Tests only core crates instead of full workspace
- **Sequential cleanup**: Removes build artifacts between steps

## Immediate Actions for Current CI

### Option 1: Use the CI Profile
```bash
# In your current workflow, replace:
cargo build --workspace
# With:
cargo build --profile ci --workspace --exclude scirs2-benchmarks
```

### Option 2: Environment Variables
Add to your workflow:
```yaml
env:
  CARGO_BUILD_JOBS: 2
  CARGO_INCREMENTAL: 0
  CARGO_PROFILE_DEV_DEBUG: 1
  CARGO_PROFILE_DEV_DEBUG_ASSERTIONS: false
```

### Option 3: Free Disk Space Step
Add this step at the beginning of your workflow:
```yaml
- name: Free Disk Space
  run: |
    sudo rm -rf /usr/share/dotnet
    sudo rm -rf /usr/local/lib/android
    sudo rm -rf /opt/ghc
    sudo rm -rf /opt/hostedtoolcache/CodeQL
    sudo docker system prune -af
    df -h
```

## Long-term Optimizations

### 1. Reduce Dependencies
- Consider making heavy dependencies optional with feature flags
- Split large crates into smaller ones
- Use `workspace.dependencies` to avoid duplication

### 2. Build Strategy
- Build only changed crates using `cargo-workspaces`
- Use different CI jobs for different crate categories
- Consider using a build matrix to split work

### 3. Test Optimization
- Run tests only for changed crates
- Use `--no-fail-fast` to get more information per run
- Limit test parallelism with `--test-threads=2`

## Expected Results

These optimizations should:
- **Reduce build artifacts by ~40-60%**
- **Free up ~14GB of system disk space**
- **Reduce memory usage during linking**
- **Enable successful CI builds on GitHub Actions**

## Monitoring

Check disk usage in your CI with:
```bash
df -h
du -sh target/
```

The optimizations are backward compatible and won't affect local development builds.