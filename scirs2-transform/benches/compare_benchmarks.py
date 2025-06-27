#!/usr/bin/env python3
"""
Compare benchmark results between scirs2-transform and scikit-learn.

This script parses benchmark outputs from both Rust and Python implementations
and generates a comparison report.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess


def parse_criterion_output(output: str) -> Dict[str, Dict[str, float]]:
    """Parse Criterion benchmark output."""
    results = {}
    
    # Pattern to match Criterion output lines
    # Example: "MinMax/100x10        time:   [1.2345 ms 1.2456 ms 1.2567 ms]"
    pattern = r'(\w+)/(\d+x\d+)\s+time:\s+\[[\d.]+ \w+ ([\d.]+) (\w+) [\d.]+ \w+\]'
    
    for match in re.finditer(pattern, output):
        bench_name = match.group(1)
        size = match.group(2)
        time_value = float(match.group(3))
        time_unit = match.group(4)
        
        # Convert to milliseconds
        if time_unit == 'us' or time_unit == 'µs':
            time_ms = time_value / 1000
        elif time_unit == 'ms':
            time_ms = time_value
        elif time_unit == 's':
            time_ms = time_value * 1000
        else:
            time_ms = time_value
        
        if bench_name not in results:
            results[bench_name] = {}
        results[bench_name][size] = time_ms
    
    return results


def parse_sklearn_output(output: str) -> Dict[str, Dict[str, float]]:
    """Parse scikit-learn benchmark output."""
    results = {}
    current_size = None
    
    for line in output.split('\n'):
        # Match data shape lines
        size_match = re.match(r'Data shape: (\d+x\d+)', line)
        if size_match:
            current_size = size_match.group(1)
            continue
        
        # Match benchmark result lines
        # Example: "MinMax: 1.234±0.123 ms, Throughput: 1.23 M elements/s"
        result_match = re.match(r'(\w+): ([\d.]+)±[\d.]+ ms', line)
        if result_match and current_size:
            bench_name = result_match.group(1)
            time_ms = float(result_match.group(2))
            
            if bench_name not in results:
                results[bench_name] = {}
            results[bench_name][current_size] = time_ms
    
    return results


def generate_comparison_table(rust_results: Dict, sklearn_results: Dict) -> str:
    """Generate a markdown comparison table."""
    output = []
    output.append("# Performance Comparison: scirs2-transform vs scikit-learn\n")
    
    # Get all benchmark names and sizes
    all_benchmarks = set(rust_results.keys()) | set(sklearn_results.keys())
    all_sizes = set()
    for results in [rust_results, sklearn_results]:
        for bench_results in results.values():
            all_sizes.update(bench_results.keys())
    
    all_sizes = sorted(all_sizes, key=lambda x: tuple(map(int, x.split('x'))))
    
    for benchmark in sorted(all_benchmarks):
        output.append(f"\n## {benchmark}\n")
        output.append("| Data Size | scirs2 (ms) | sklearn (ms) | Speedup |")
        output.append("|-----------|-------------|--------------|---------|")
        
        for size in all_sizes:
            rust_time = rust_results.get(benchmark, {}).get(size, None)
            sklearn_time = sklearn_results.get(benchmark, {}).get(size, None)
            
            if rust_time is not None and sklearn_time is not None:
                speedup = sklearn_time / rust_time
                output.append(
                    f"| {size} | {rust_time:.3f} | {sklearn_time:.3f} | "
                    f"{speedup:.2f}x |"
                )
            elif rust_time is not None:
                output.append(f"| {size} | {rust_time:.3f} | - | - |")
            elif sklearn_time is not None:
                output.append(f"| {size} | - | {sklearn_time:.3f} | - |")
    
    # Add summary statistics
    output.append("\n## Summary Statistics\n")
    
    total_rust = 0
    total_sklearn = 0
    count = 0
    speedups = []
    
    for benchmark in all_benchmarks:
        for size in all_sizes:
            rust_time = rust_results.get(benchmark, {}).get(size, None)
            sklearn_time = sklearn_results.get(benchmark, {}).get(size, None)
            
            if rust_time is not None and sklearn_time is not None:
                total_rust += rust_time
                total_sklearn += sklearn_time
                speedups.append(sklearn_time / rust_time)
                count += 1
    
    if count > 0:
        avg_speedup = sum(speedups) / len(speedups)
        min_speedup = min(speedups)
        max_speedup = max(speedups)
        
        output.append(f"- **Average speedup**: {avg_speedup:.2f}x")
        output.append(f"- **Min speedup**: {min_speedup:.2f}x")
        output.append(f"- **Max speedup**: {max_speedup:.2f}x")
        output.append(f"- **Total benchmarks compared**: {count}")
    
    return '\n'.join(output)


def run_rust_benchmarks() -> str:
    """Run Rust benchmarks and return output."""
    print("Running Rust benchmarks...")
    try:
        result = subprocess.run(
            ['cargo', 'bench', '--bench', 'transform_bench'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running Rust benchmarks: {e}")
        return ""


def run_sklearn_benchmarks() -> str:
    """Run scikit-learn benchmarks and return output."""
    print("Running scikit-learn benchmarks...")
    try:
        result = subprocess.run(
            ['python', 'benches/sklearn_comparison.py'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running scikit-learn benchmarks: {e}")
        return ""


def main():
    """Main function to run comparisons."""
    # Check if we should run benchmarks or use existing results
    if len(sys.argv) > 1 and sys.argv[1] == '--run':
        rust_output = run_rust_benchmarks()
        sklearn_output = run_sklearn_benchmarks()
        
        # Save outputs for later analysis
        with open('rust_benchmark_output.txt', 'w') as f:
            f.write(rust_output)
        with open('sklearn_benchmark_output.txt', 'w') as f:
            f.write(sklearn_output)
    else:
        # Try to load existing results
        try:
            with open('rust_benchmark_output.txt', 'r') as f:
                rust_output = f.read()
            with open('sklearn_benchmark_output.txt', 'r') as f:
                sklearn_output = f.read()
        except FileNotFoundError:
            print("No benchmark results found. Run with --run flag to generate.")
            return
    
    # Parse results
    rust_results = parse_criterion_output(rust_output)
    sklearn_results = parse_sklearn_output(sklearn_output)
    
    # Generate comparison
    comparison = generate_comparison_table(rust_results, sklearn_results)
    
    # Save comparison
    with open('benchmark_comparison.md', 'w') as f:
        f.write(comparison)
    
    print("Comparison report saved to benchmark_comparison.md")
    print("\nSummary:")
    print(comparison.split("## Summary Statistics")[1] if "## Summary Statistics" in comparison else "No summary available")


if __name__ == "__main__":
    main()