#!/usr/bin/env python3
"""
Comprehensive three-way performance comparison: scirs2-graph vs NetworkX vs igraph

This script orchestrates benchmarks for all three libraries and generates
detailed performance comparisons with visualizations and statistical analysis.

Requirements:
    pip install networkx igraph numpy pandas matplotlib seaborn tabulate scipy
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
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import benchmark functions
from networkx_comparison import (
    benchmark_graph_creation as nx_creation,
    benchmark_traversal as nx_traversal,
    benchmark_shortest_paths as nx_paths,
    benchmark_connectivity as nx_connectivity,
    benchmark_centrality as nx_centrality,
    benchmark_mst as nx_mst
)

from igraph_comparison import (
    benchmark_graph_creation as ig_creation,
    benchmark_traversal as ig_traversal,
    benchmark_shortest_paths as ig_paths,
    benchmark_connectivity as ig_connectivity,
    benchmark_centrality as ig_centrality,
    benchmark_mst as ig_mst,
    benchmark_clustering as ig_clustering,
    benchmark_graph_properties as ig_properties
)


def run_rust_benchmarks() -> Dict[str, Any]:
    """Run Rust benchmarks using criterion and parse results."""
    print("🦀 Running Rust benchmarks...")
    
    # Run cargo bench and capture output
    result = subprocess.run(
        ['cargo', 'bench', '--bench', 'graph_benchmarks'],
        capture_output=True,
        text=True,
        cwd='..'
    )
    
    if result.returncode != 0:
        print(f"Warning: Rust benchmarks failed: {result.stderr}")
        return {}
    
    # Parse criterion output (simplified)
    rust_results = {}
    
    # Read the criterion benchmark results
    criterion_dir = Path('../target/criterion')
    if criterion_dir.exists():
        for bench_dir in criterion_dir.iterdir():
            if bench_dir.is_dir():
                for size_dir in bench_dir.iterdir():
                    if size_dir.is_dir():
                        estimates_file = size_dir / 'estimates.json'
                        if estimates_file.exists():
                            try:
                                with open(estimates_file, 'r') as f:
                                    data = json.load(f)
                                    bench_name = bench_dir.name
                                    size_info = size_dir.name
                                    
                                    if bench_name not in rust_results:
                                        rust_results[bench_name] = {}
                                    
                                    # Extract size from directory name
                                    size = extract_size_from_path(size_info)
                                    if size:
                                        # Convert nanoseconds to seconds
                                        rust_results[bench_name][str(size)] = data['mean']['point_estimate'] / 1e9
                            except (json.JSONDecodeError, KeyError) as e:
                                continue
    
    return rust_results


def extract_size_from_path(path_str: str) -> int:
    """Extract size number from criterion path string."""
    import re
    match = re.search(r'(\d+)', path_str)
    return int(match.group(1)) if match else None


def run_networkx_benchmarks() -> Dict[str, Any]:
    """Run comprehensive NetworkX benchmarks."""
    print("🐍 Running NetworkX benchmarks...")
    
    results = {}
    
    # Define test sizes
    test_configs = {
        'creation': {'sizes': [100, 1000, 10000], 'func': nx_creation},
        'traversal': {'sizes': [100, 1000, 10000], 'func': nx_traversal},
        'shortest_paths': {'sizes': [100, 500, 1000], 'func': nx_paths},
        'connectivity': {'sizes': [100, 1000, 5000], 'func': nx_connectivity},
        'centrality': {'sizes': [50, 100, 200], 'func': nx_centrality},
        'minimum_spanning_tree': {'sizes': [100, 500, 1000], 'func': nx_mst}
    }
    
    for category, config in test_configs.items():
        print(f"  Running {category}...")
        try:
            category_results = config['func'](config['sizes'])
            results.update(category_results)
        except Exception as e:
            print(f"  Warning: {category} failed: {e}")
    
    return results


def run_igraph_benchmarks() -> Dict[str, Any]:
    """Run comprehensive igraph benchmarks."""
    print("📊 Running igraph benchmarks...")
    
    results = {}
    
    # Define test sizes
    test_configs = {
        'creation': {'sizes': [100, 1000, 10000], 'func': ig_creation},
        'traversal': {'sizes': [100, 1000, 10000], 'func': ig_traversal},
        'shortest_paths': {'sizes': [100, 500, 1000], 'func': ig_paths},
        'connectivity': {'sizes': [100, 1000, 5000], 'func': ig_connectivity},
        'centrality': {'sizes': [50, 100, 200], 'func': ig_centrality},
        'minimum_spanning_tree': {'sizes': [100, 500, 1000], 'func': ig_mst},
        'clustering': {'sizes': [50, 100, 200], 'func': ig_clustering},
        'properties': {'sizes': [100, 500, 1000], 'func': ig_properties}
    }
    
    for category, config in test_configs.items():
        print(f"  Running {category}...")
        try:
            category_results = config['func'](config['sizes'])
            results.update(category_results)
        except Exception as e:
            print(f"  Warning: {category} failed: {e}")
    
    return results


def calculate_speedup(base_time: float, compare_time: float) -> float:
    """Calculate speedup factor."""
    if base_time == 0:
        return float('inf')
    return compare_time / base_time


def generate_comprehensive_report(rust_results: Dict, networkx_results: Dict, igraph_results: Dict):
    """Generate comprehensive three-way performance comparison report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Collect all benchmark data
    comparison_data = []
    
    # Create unified dataset for analysis
    all_benchmarks = set()
    all_benchmarks.update(rust_results.keys())
    all_benchmarks.update(networkx_results.keys())
    all_benchmarks.update(igraph_results.keys())
    
    for benchmark in all_benchmarks:
        # Get all sizes for this benchmark across libraries
        all_sizes = set()
        if benchmark in rust_results:
            all_sizes.update(rust_results[benchmark].keys())
        if benchmark in networkx_results:
            all_sizes.update(str(s) for s in networkx_results[benchmark].keys())
        if benchmark in igraph_results:
            all_sizes.update(str(s) for s in igraph_results[benchmark].keys())
        
        for size_str in all_sizes:
            size = int(size_str)
            row = {
                'Benchmark': benchmark,
                'Size': size,
                'Rust': None,
                'NetworkX': None,
                'igraph': None
            }
            
            # Get Rust time
            if benchmark in rust_results and size_str in rust_results[benchmark]:
                row['Rust'] = rust_results[benchmark][size_str]
            
            # Get NetworkX time
            if benchmark in networkx_results and size in networkx_results[benchmark]:
                row['NetworkX'] = networkx_results[benchmark][size]
            
            # Get igraph time
            if benchmark in igraph_results and size in igraph_results[benchmark]:
                row['igraph'] = igraph_results[benchmark][size]
            
            # Only add row if we have at least 2 libraries for comparison
            valid_times = sum(1 for t in [row['Rust'], row['NetworkX'], row['igraph']] if t is not None)
            if valid_times >= 2:
                comparison_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    df = df.dropna(thresh=3)  # Keep rows with at least 2 non-null values
    
    if df.empty:
        print("Warning: No comparable benchmarks found between libraries")
        return df
    
    # Calculate speedups
    for idx, row in df.iterrows():
        if row['Rust'] is not None and row['NetworkX'] is not None:
            df.loc[idx, 'Rust_vs_NetworkX'] = calculate_speedup(row['Rust'], row['NetworkX'])
        
        if row['Rust'] is not None and row['igraph'] is not None:
            df.loc[idx, 'Rust_vs_igraph'] = calculate_speedup(row['Rust'], row['igraph'])
        
        if row['NetworkX'] is not None and row['igraph'] is not None:
            df.loc[idx, 'NetworkX_vs_igraph'] = calculate_speedup(row['NetworkX'], row['igraph'])
    
    # Generate markdown report
    report = f"""
# Three-Way Performance Comparison: scirs2-graph vs NetworkX vs igraph
Generated: {timestamp}

## Executive Summary

This report compares the performance of scirs2-graph (Rust), NetworkX (Python), and igraph (Python/C)
across various graph algorithms and operations.

## Performance Overview

"""
    
    # Summary statistics
    if not df[['Rust_vs_NetworkX', 'Rust_vs_igraph', 'NetworkX_vs_igraph']].empty:
        speedup_summary = df[['Rust_vs_NetworkX', 'Rust_vs_igraph', 'NetworkX_vs_igraph']].describe()
        report += "### Speedup Statistics\n\n"
        report += tabulate(speedup_summary, headers=['', 'Rust vs NetworkX', 'Rust vs igraph', 'NetworkX vs igraph'], 
                          tablefmt='markdown', floatfmt=".2f")
        report += "\n\n"
    
    # Detailed comparison
    report += "### Detailed Performance Comparison\n\n"
    display_df = df[['Benchmark', 'Size', 'Rust', 'NetworkX', 'igraph', 'Rust_vs_NetworkX', 'Rust_vs_igraph']].copy()
    report += tabulate(display_df, headers='keys', tablefmt='markdown', floatfmt=".6f")
    report += "\n\n"
    
    # Performance insights
    report += "## Performance Insights\n\n"
    
    # Best and worst performers
    speedup_cols = ['Rust_vs_NetworkX', 'Rust_vs_igraph', 'NetworkX_vs_igraph']
    for col in speedup_cols:
        if col in df.columns and not df[col].isna().all():
            best = df.loc[df[col].idxmax()]
            worst = df.loc[df[col].idxmin()]
            lib_comparison = col.replace('_vs_', ' vs ')
            
            report += f"### {lib_comparison}\n"
            report += f"- **Best speedup**: {best['Benchmark']} (size {best['Size']}) - {best[col]:.2f}x\n"
            report += f"- **Worst speedup**: {worst['Benchmark']} (size {worst['Size']}) - {worst[col]:.2f}x\n"
            report += f"- **Average speedup**: {df[col].mean():.2f}x\n\n"
    
    # Algorithm ranking
    report += "## Algorithm Performance Ranking\n\n"
    algo_performance = df.groupby('Benchmark')[['Rust', 'NetworkX', 'igraph']].mean()
    for algo in algo_performance.index:
        report += f"### {algo}\n"
        algo_times = algo_performance.loc[algo].dropna().sort_values()
        for i, (lib, time_val) in enumerate(algo_times.items(), 1):
            report += f"{i}. {lib}: {time_val:.6f}s (avg)\n"
        report += "\n"
    
    # Save report
    with open('three_way_performance_comparison.md', 'w') as f:
        f.write(report)
    
    return df


def create_comprehensive_visualizations(df: pd.DataFrame):
    """Create comprehensive performance comparison visualizations."""
    if df.empty:
        print("No data to visualize")
        return
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # 1. Performance comparison heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for heatmap
    pivot_data = df.pivot_table(
        index='Benchmark', 
        columns='Size', 
        values=['Rust', 'NetworkX', 'igraph'],
        aggfunc='mean'
    )
    
    if not pivot_data.empty:
        # Normalize data for better visualization (log scale)
        pivot_normalized = np.log10(pivot_data + 1e-10)  # Add small value to avoid log(0)
        
        sns.heatmap(pivot_normalized, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
        ax.set_title('Performance Heatmap (log10 scale)', fontsize=16)
        ax.set_xlabel('Graph Size', fontsize=12)
        ax.set_ylabel('Algorithm', fontsize=12)
        plt.tight_layout()
        plt.savefig('three_way_performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Speedup comparison chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    speedup_cols = ['Rust_vs_NetworkX', 'Rust_vs_igraph', 'NetworkX_vs_igraph']
    titles = ['Rust vs NetworkX', 'Rust vs igraph', 'NetworkX vs igraph']
    
    for i, (col, title) in enumerate(zip(speedup_cols, titles)):
        if col in df.columns and not df[col].isna().all():
            speedup_data = df.groupby('Benchmark')[col].mean().sort_values(ascending=False)
            
            axes[i].bar(range(len(speedup_data)), speedup_data.values)
            axes[i].set_title(f'Average Speedup: {title}')
            axes[i].set_xlabel('Algorithm')
            axes[i].set_ylabel('Speedup Factor')
            axes[i].set_xticks(range(len(speedup_data)))
            axes[i].set_xticklabels(speedup_data.index, rotation=45, ha='right')
            axes[i].axhline(y=1, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('three_way_speedup_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Scaling behavior comparison
    algorithms = df['Benchmark'].unique()[:6]  # Top 6 algorithms
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = {'Rust': '#1f77b4', 'NetworkX': '#ff7f0e', 'igraph': '#2ca02c'}
    
    for idx, algo in enumerate(algorithms):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        algo_data = df[df['Benchmark'] == algo].sort_values('Size')
        
        for lib in ['Rust', 'NetworkX', 'igraph']:
            if lib in algo_data.columns:
                valid_data = algo_data[algo_data[lib].notna()]
                if not valid_data.empty:
                    ax.plot(valid_data['Size'], valid_data[lib], 'o-', 
                           label=lib, color=colors[lib], linewidth=2, markersize=6)
        
        ax.set_xlabel('Graph Size')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(algo.replace('_', ' ').title())
        ax.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Scaling Behavior: Three-Way Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig('three_way_scaling_behavior.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Performance distribution
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Box plot of execution times by library
    plot_data = []
    for lib in ['Rust', 'NetworkX', 'igraph']:
        if lib in df.columns:
            lib_times = df[df[lib].notna()][lib].values
            plot_data.extend([(lib, time) for time in lib_times])
    
    if plot_data:
        plot_df = pd.DataFrame(plot_data, columns=['Library', 'Time'])
        sns.boxplot(data=plot_df, x='Library', y='Time', ax=ax)
        ax.set_yscale('log')
        ax.set_title('Performance Distribution by Library', fontsize=16)
        ax.set_ylabel('Execution Time (seconds, log scale)', fontsize=12)
        plt.tight_layout()
        plt.savefig('three_way_performance_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()


def statistical_analysis(df: pd.DataFrame):
    """Perform statistical analysis of performance differences."""
    if df.empty:
        return
    
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    
    # Pairwise comparisons
    comparisons = [
        ('Rust', 'NetworkX'),
        ('Rust', 'igraph'),
        ('NetworkX', 'igraph')
    ]
    
    for lib1, lib2 in comparisons:
        if lib1 in df.columns and lib2 in df.columns:
            # Get paired samples (same benchmark, same size)
            paired_data = df[[lib1, lib2]].dropna()
            
            if len(paired_data) > 1:
                # Wilcoxon signed-rank test for paired samples
                statistic, p_value = stats.wilcoxon(
                    paired_data[lib1], 
                    paired_data[lib2],
                    alternative='two-sided'
                )
                
                print(f"\n{lib1} vs {lib2}:")
                print(f"  Paired samples: {len(paired_data)}")
                print(f"  Wilcoxon statistic: {statistic:.4f}")
                print(f"  P-value: {p_value:.6f}")
                print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
                
                # Effect size (median difference)
                median_diff = paired_data[lib1].median() - paired_data[lib2].median()
                print(f"  Median difference: {median_diff:.6f}s")
                
                faster_lib = lib1 if median_diff < 0 else lib2
                print(f"  Faster library: {faster_lib}")


def main():
    """Run comprehensive three-way performance comparison."""
    print("🚀 Starting comprehensive three-way performance comparison...")
    print("="*80)
    
    # Run all benchmarks
    rust_results = run_rust_benchmarks()
    networkx_results = run_networkx_benchmarks()
    igraph_results = run_igraph_benchmarks()
    
    # Save raw results
    results_data = {
        'rust': rust_results,
        'networkx': networkx_results,
        'igraph': igraph_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('three_way_benchmark_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Generate comparison report
    print("\n📊 Generating comprehensive performance report...")
    df = generate_comprehensive_report(rust_results, networkx_results, igraph_results)
    
    if not df.empty:
        # Create visualizations
        print("📈 Creating visualizations...")
        create_comprehensive_visualizations(df)
        
        # Statistical analysis
        statistical_analysis(df)
        
        # Print summary
        print("\n" + "="*80)
        print("THREE-WAY PERFORMANCE COMPARISON SUMMARY")
        print("="*80)
        
        # Overall performance ranking
        avg_times = df[['Rust', 'NetworkX', 'igraph']].mean()
        print(f"\nAverage execution times:")
        for lib, avg_time in avg_times.sort_values().items():
            print(f"  {lib}: {avg_time:.6f}s")
        
        # Speedup summaries
        speedup_cols = ['Rust_vs_NetworkX', 'Rust_vs_igraph', 'NetworkX_vs_igraph']
        print(f"\nSpeedup factors:")
        for col in speedup_cols:
            if col in df.columns and not df[col].isna().all():
                avg_speedup = df[col].mean()
                comparison = col.replace('_vs_', ' vs ')
                print(f"  {comparison}: {avg_speedup:.2f}x average")
        
        print("\nFiles generated:")
        print("  - three_way_performance_comparison.md")
        print("  - three_way_performance_heatmap.png")
        print("  - three_way_speedup_comparison.png")
        print("  - three_way_scaling_behavior.png")
        print("  - three_way_performance_distribution.png")
        print("  - three_way_benchmark_results.json")
        
    else:
        print("❌ No comparable benchmarks found between libraries")
    
    print("\n✅ Three-way comparison completed!")


if __name__ == "__main__":
    main()