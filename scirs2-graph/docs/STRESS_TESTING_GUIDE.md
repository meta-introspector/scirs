# Stress Testing Guide for SciRS2-Graph

This guide explains how to run and interpret stress tests for scirs2-graph with large graphs (>1M nodes).

## Running Stress Tests

The stress tests are located in `tests/stress_tests.rs` and are marked with `#[ignore]` to prevent them from running during normal test cycles.

### Basic Command

```bash
# Run all stress tests with single thread (recommended for accurate timing)
cargo test stress_tests -- --ignored --test-threads=1 --nocapture

# Run specific stress test
cargo test test_large_erdos_renyi_graph -- --ignored --nocapture
```

### With Performance Features

```bash
# Enable all performance features
cargo test stress_tests --features "parallel simd" -- --ignored --test-threads=1 --nocapture

# Run in release mode for accurate performance measurements
cargo test --release stress_tests -- --ignored --test-threads=1 --nocapture
```

## Test Categories

### 1. Graph Generation Tests

Tests the performance of generating various types of large graphs:

- **Erdős-Rényi graphs**: Random graphs with specified edge probability
- **Barabási-Albert graphs**: Scale-free networks with preferential attachment
- **Grid graphs**: Regular 2D lattice structures
- **Directed graphs**: Large directed graphs with various patterns

### 2. Algorithm Scaling Tests

Measures how algorithm performance scales with graph size:

```
Graph Size | BFS Time | Connected Components | PageRank
-----------|----------|---------------------|----------
10,000     | 0.012s   | 0.015s             | 0.023s
50,000     | 0.065s   | 0.082s             | 0.125s
100,000    | 0.134s   | 0.171s             | 0.256s
200,000    | 0.278s   | 0.352s             | 0.523s
```

### 3. Memory Efficiency Tests

Tests operations on very large graphs (1M+ nodes) with limited memory:

- Streaming degree calculations
- Sample-based clustering coefficients
- Depth-limited traversals

### 4. Extreme Scale Tests

Pushes the limits with 5M+ node graphs to identify bottlenecks.

## Interpreting Results

### Performance Metrics

1. **Generation Time**: How long it takes to create the graph
   - Should scale linearly with number of edges
   - Barabási-Albert should be O(n*m) where m is edges per node

2. **Algorithm Time**: Execution time for various algorithms
   - BFS/DFS: Should be O(V + E)
   - Connected Components: Should be O(V + E)
   - PageRank: Should be O(k*(V + E)) for k iterations

3. **Memory Usage**: Estimated memory consumption
   - Adjacency list: ~16 bytes per edge + overhead
   - Typical: 100MB per million edges

### Expected Performance

For a modern machine (8+ cores, 16GB+ RAM):

| Graph Size | Type | Generation | Basic Ops | Memory |
|------------|------|------------|-----------|---------|
| 100K nodes | Sparse | <1s | <0.1s | ~10MB |
| 1M nodes | Sparse | <10s | <1s | ~100MB |
| 5M nodes | Sparse | <60s | <5s | ~500MB |
| 1M nodes | Dense | <30s | <10s | ~1GB |

### Bottleneck Identification

Look for:

1. **Superlinear scaling**: Algorithm time growing faster than expected
2. **Memory spikes**: Sudden increases in memory usage
3. **Cache misses**: Performance degradation at certain sizes
4. **Thread contention**: Parallel performance not scaling linearly

## Profiling Large Graphs

### Using Valgrind/Massif

```bash
# Memory profiling
valgrind --tool=massif --massif-out-file=massif.out \
    cargo test --release test_large_barabasi_albert_graph -- --ignored

# Analyze results
ms_print massif.out
```

### Using perf

```bash
# CPU profiling
perf record cargo test --release test_large_erdos_renyi_graph -- --ignored
perf report
```

### Using cargo-flamegraph

```bash
# Install flamegraph
cargo install flamegraph

# Generate flamegraph
cargo flamegraph --test stress_tests -- test_large_grid_graph --ignored
```

## Common Issues and Solutions

### Out of Memory

**Problem**: Tests fail with allocation errors on very large graphs.

**Solutions**:
1. Reduce graph density (lower edge probability)
2. Use streaming/chunked algorithms
3. Enable swap space temporarily
4. Use a machine with more RAM

### Slow Performance

**Problem**: Tests take too long to complete.

**Solutions**:
1. Run in release mode: `cargo test --release`
2. Enable parallel features: `--features parallel`
3. Reduce number of iterations for iterative algorithms
4. Use sampling for approximate results

### Stack Overflow

**Problem**: Deep recursion in algorithms causes stack overflow.

**Solutions**:
1. Increase stack size: `RUST_MIN_STACK=8388608 cargo test`
2. Use iterative instead of recursive implementations
3. Use bounded depth algorithms

## Benchmarking Best Practices

### 1. Consistent Environment

```bash
# Disable CPU frequency scaling
sudo cpupower frequency-set -g performance

# Pin to specific CPU cores
taskset -c 0-3 cargo test --release
```

### 2. Multiple Runs

```bash
# Run tests multiple times and average results
for i in {1..5}; do
    cargo test --release test_algorithm_scaling -- --ignored --nocapture
done | grep "time:" | awk '{sum+=$3; count++} END {print "Average:", sum/count}'
```

### 3. Memory Monitoring

```bash
# Monitor memory usage during tests
watch -n 1 'ps aux | grep stress_tests'

# Or use /usr/bin/time for summary
/usr/bin/time -v cargo test --release stress_tests -- --ignored
```

## Continuous Performance Testing

### Integration with CI

```yaml
# .github/workflows/perf-tests.yml
name: Performance Tests

on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday
  workflow_dispatch:

jobs:
  stress-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Run stress tests
        run: |
          cargo test --release stress_tests -- --ignored --test-threads=1
        timeout-minutes: 60
        
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: stress-test-results
          path: target/stress-test-*.log
```

### Performance Regression Detection

Create a baseline:

```bash
# Generate baseline
cargo test --release test_algorithm_scaling -- --ignored > baseline.txt

# Compare with new results
cargo test --release test_algorithm_scaling -- --ignored > current.txt
diff baseline.txt current.txt
```

## Customizing Stress Tests

### Adding New Graph Sizes

```rust
// In stress_tests.rs
let sizes = vec![100_000, 500_000, 1_000_000, 2_000_000]; // Add 2M
```

### Testing Specific Algorithms

```rust
#[test]
#[ignore]
fn test_my_algorithm_at_scale() -> CoreResult<()> {
    let n = 1_000_000;
    let graph = generators::erdos_renyi_graph(n, 0.00001, None)?;
    
    let start = Instant::now();
    let result = my_algorithm(&graph)?;
    println!("My algorithm on {}M nodes: {:.2}s", 
        n/1_000_000, start.elapsed().as_secs_f64());
    
    Ok(())
}
```

### Memory-Limited Testing

```rust
// Test with constrained memory
const MEMORY_LIMIT_MB: usize = 512;

fn can_fit_in_memory(nodes: usize, edges: usize) -> bool {
    let estimated_mb = estimate_memory_usage(nodes, edges);
    estimated_mb < MEMORY_LIMIT_MB as f64
}
```

## Reporting Issues

When reporting performance issues with large graphs, include:

1. **System specs**: CPU, RAM, OS
2. **Graph details**: Size, type, density
3. **Timing results**: From stress tests
4. **Memory usage**: Peak and average
5. **Cargo features**: Which features were enabled
6. **Comparison**: With expected performance

Example issue report:

```
Title: PageRank slow on 1M node graphs

System: Intel i7-9700K, 32GB RAM, Ubuntu 22.04
Graph: Barabási-Albert, 1M nodes, m=5 (5M edges)
Features: parallel, simd

Results:
- Generation: 8.2s (expected: <10s) ✓
- PageRank (10 iter): 45s (expected: <10s) ✗
- Memory peak: 890MB

The PageRank implementation seems to have quadratic scaling...
```