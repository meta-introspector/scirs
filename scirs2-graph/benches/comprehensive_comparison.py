#!/usr/bin/env python3
"""
Comprehensive performance comparison between scirs2-graph and NetworkX

This script orchestrates benchmarks for both libraries and generates
detailed performance comparisons with visualizations and analysis.

Requirements:
    pip install networkx numpy pandas matplotlib seaborn tabulate
"""

import subprocess
import json
import time
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
from tabulate import tabulate
import networkx as nx
from datetime import datetime

# Import the NetworkX benchmark functions
from networkx_comparison import (
    benchmark_graph_creation,
    benchmark_traversal,
    benchmark_shortest_paths,
    benchmark_connectivity,
    benchmark_centrality,
    benchmark_mst
)

def run_rust_benchmarks() -> Dict[str, Any]:
    """Run Rust benchmarks using criterion and parse results."""
    print("Running Rust benchmarks...")
    
    # Run cargo bench and capture output
    result = subprocess.run(
        ['cargo', 'bench', '--bench', 'graph_benchmarks', '--', '--output-format', 'json'],
        capture_output=True,
        text=True,
        cwd='..'  # Run from parent directory
    )
    
    if result.returncode != 0:
        print(f"Error running Rust benchmarks: {result.stderr}")
        return {}
    
    # Parse criterion output
    rust_results = {}
    
    # Read the criterion benchmark results
    criterion_dir = Path('../target/criterion')
    if criterion_dir.exists():
        for bench_dir in criterion_dir.iterdir():
            if bench_dir.is_dir():
                # Parse each benchmark group
                for size_dir in bench_dir.iterdir():
                    if size_dir.is_dir() and 'base' in str(size_dir):
                        estimates_file = size_dir / 'estimates.json'
                        if estimates_file.exists():
                            with open(estimates_file, 'r') as f:
                                data = json.load(f)
                                bench_name = bench_dir.name
                                size = size_dir.name.split('/')[-1]
                                
                                if bench_name not in rust_results:
                                    rust_results[bench_name] = {}
                                
                                # Convert nanoseconds to seconds
                                rust_results[bench_name][size] = data['mean']['point_estimate'] / 1e9
    
    return rust_results

def run_comprehensive_networkx_benchmarks() -> Dict[str, Any]:
    """Run comprehensive NetworkX benchmarks."""
    print("\nRunning NetworkX benchmarks...")
    
    results = {}
    
    # Define test sizes matching Rust benchmarks
    test_configs = {
        'graph_creation': {
            'sizes': [100, 1000, 10000],
            'func': benchmark_graph_creation
        },
        'traversal': {
            'sizes': [100, 1000, 10000],
            'func': benchmark_traversal
        },
        'shortest_paths': {
            'sizes': [100, 500, 1000],
            'func': benchmark_shortest_paths
        },
        'connectivity': {
            'sizes': [100, 1000, 5000],
            'func': benchmark_connectivity
        },
        'centrality': {
            'sizes': [50, 100, 200],
            'func': benchmark_centrality
        },
        'minimum_spanning_tree': {
            'sizes': [100, 500, 1000],
            'func': benchmark_mst
        }
    }
    
    for category, config in test_configs.items():
        print(f"  Running {category}...")
        category_results = config['func'](config['sizes'])
        results.update(category_results)
    
    return results

def calculate_speedup(rust_time: float, python_time: float) -> float:
    """Calculate speedup factor of Rust over Python."""
    if rust_time == 0:
        return float('inf')
    return python_time / rust_time

def generate_performance_report(rust_results: Dict, python_results: Dict):
    """Generate comprehensive performance comparison report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
# Performance Comparison Report: scirs2-graph vs NetworkX
Generated: {timestamp}

## Executive Summary

This report compares the performance of scirs2-graph (Rust) with NetworkX (Python)
across various graph algorithms and operations.

## Detailed Results

"""
    
    # Collect all benchmark data
    comparison_data = []
    
    for benchmark in python_results:
        if benchmark in rust_results:
            for size in python_results[benchmark]:
                if str(size) in rust_results[benchmark]:
                    rust_time = rust_results[benchmark][str(size)]
                    python_time = python_results[benchmark][size]
                    speedup = calculate_speedup(rust_time, python_time)
                    
                    comparison_data.append({
                        'Benchmark': benchmark,
                        'Size': size,
                        'Rust (s)': rust_time,
                        'Python (s)': python_time,
                        'Speedup': speedup
                    })
    
    # Create DataFrame for analysis
    df = pd.DataFrame(comparison_data)
    
    # Summary statistics by benchmark
    summary = df.groupby('Benchmark')['Speedup'].agg(['mean', 'min', 'max'])
    
    report += "### Average Speedup by Algorithm\n\n"
    report += tabulate(summary, headers=['Algorithm', 'Mean Speedup', 'Min Speedup', 'Max Speedup'], 
                      tablefmt='markdown')
    report += "\n\n"
    
    # Detailed comparison table
    report += "### Detailed Performance Comparison\n\n"
    report += tabulate(df, headers='keys', tablefmt='markdown', floatfmt=".6f")
    report += "\n\n"
    
    # Performance insights
    report += "## Performance Insights\n\n"
    
    best_speedup = df.loc[df['Speedup'].idxmax()]
    worst_speedup = df.loc[df['Speedup'].idxmin()]
    
    report += f"- **Best speedup**: {best_speedup['Benchmark']} (size {best_speedup['Size']}) - "
    report += f"{best_speedup['Speedup']:.2f}x faster\n"
    report += f"- **Worst speedup**: {worst_speedup['Benchmark']} (size {worst_speedup['Size']}) - "
    report += f"{worst_speedup['Speedup']:.2f}x faster\n"
    report += f"- **Average speedup across all benchmarks**: {df['Speedup'].mean():.2f}x\n"
    
    # Save report
    with open('performance_comparison_report.md', 'w') as f:
        f.write(report)
    
    return df

def create_visualizations(df: pd.DataFrame):
    """Create performance comparison visualizations."""
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Speedup by algorithm
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for grouped bar chart
    pivot_df = df.pivot(index='Benchmark', columns='Size', values='Speedup')
    pivot_df.plot(kind='bar', ax=ax)
    
    ax.set_title('scirs2-graph Speedup over NetworkX by Algorithm and Graph Size', fontsize=16)
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Speedup Factor (x times faster)', fontsize=12)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal performance')
    ax.legend(title='Graph Size', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('speedup_by_algorithm.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Execution time comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    algorithms = df['Benchmark'].unique()[:6]  # Take first 6 algorithms
    
    for idx, algo in enumerate(algorithms):
        ax = axes[idx]
        algo_df = df[df['Benchmark'] == algo]
        
        x = np.arange(len(algo_df))
        width = 0.35
        
        rust_bars = ax.bar(x - width/2, algo_df['Rust (s)'], width, 
                           label='scirs2-graph', color='#1f77b4')
        python_bars = ax.bar(x + width/2, algo_df['Python (s)'], width, 
                             label='NetworkX', color='#ff7f0e')
        
        ax.set_xlabel('Graph Size')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(algo.replace('_', ' ').title())
        ax.set_xticks(x)
        ax.set_xticklabels(algo_df['Size'])
        ax.legend()
        ax.set_yscale('log')
        
        # Add speedup annotations
        for i, (idx_row, row) in enumerate(algo_df.iterrows()):
            ax.text(i, max(row['Rust (s)'], row['Python (s)']) * 1.1,
                   f"{row['Speedup']:.1f}x", ha='center', va='bottom')
    
    plt.suptitle('Execution Time Comparison: scirs2-graph vs NetworkX', fontsize=16)
    plt.tight_layout()
    plt.savefig('execution_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Speedup heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    heatmap_data = df.pivot(index='Benchmark', columns='Size', values='Speedup')
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Speedup Factor'}, ax=ax)
    
    ax.set_title('Performance Speedup Heatmap', fontsize=16)
    ax.set_xlabel('Graph Size', fontsize=12)
    ax.set_ylabel('Algorithm', fontsize=12)
    plt.tight_layout()
    plt.savefig('speedup_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Scaling behavior
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, algo in enumerate(algorithms):
        ax = axes[idx]
        algo_df = df[df['Benchmark'] == algo].sort_values('Size')
        
        ax.plot(algo_df['Size'], algo_df['Rust (s)'], 'o-', 
                label='scirs2-graph', linewidth=2, markersize=8)
        ax.plot(algo_df['Size'], algo_df['Python (s)'], 's-', 
                label='NetworkX', linewidth=2, markersize=8)
        
        ax.set_xlabel('Graph Size')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(algo.replace('_', ' ').title())
        ax.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Scaling Behavior: scirs2-graph vs NetworkX', fontsize=16)
    plt.tight_layout()
    plt.savefig('scaling_behavior.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Run comprehensive performance comparison."""
    print("Starting comprehensive performance comparison...")
    print("=" * 60)
    
    # Run benchmarks
    rust_results = run_rust_benchmarks()
    python_results = run_comprehensive_networkx_benchmarks()
    
    # Save raw results
    with open('rust_benchmark_results.json', 'w') as f:
        json.dump(rust_results, f, indent=2)
    
    with open('python_benchmark_results.json', 'w') as f:
        json.dump(python_results, f, indent=2)
    
    # Generate comparison report
    print("\nGenerating performance report...")
    df = generate_performance_report(rust_results, python_results)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(df)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("=" * 60)
    
    avg_speedup = df['Speedup'].mean()
    print(f"\nAverage speedup of scirs2-graph over NetworkX: {avg_speedup:.2f}x")
    
    print("\nSpeedup by algorithm category:")
    for algo in df['Benchmark'].unique():
        algo_speedup = df[df['Benchmark'] == algo]['Speedup'].mean()
        print(f"  {algo}: {algo_speedup:.2f}x faster")
    
    print("\nFiles generated:")
    print("  - performance_comparison_report.md")
    print("  - speedup_by_algorithm.png")
    print("  - execution_time_comparison.png")
    print("  - speedup_heatmap.png")
    print("  - scaling_behavior.png")
    print("  - rust_benchmark_results.json")
    print("  - python_benchmark_results.json")

if __name__ == "__main__":
    main()