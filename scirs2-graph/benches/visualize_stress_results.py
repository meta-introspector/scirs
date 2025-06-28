#!/usr/bin/env python3
"""
Visualize stress test results for scirs2-graph large graph benchmarks

This script creates visualizations from the JSON output of stress tests,
helping to understand performance characteristics and scaling behavior.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Any

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_stress_test_results(filename: str) -> Dict[str, Any]:
    """Load stress test results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def plot_generation_performance(results: Dict[str, Any], output_dir: Path):
    """Plot graph generation performance."""
    gen_data = results['graph_generation']
    
    if not gen_data:
        print("No generation data found")
        return
    
    df = pd.DataFrame(gen_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Generation time vs graph size
    ax = axes[0, 0]
    ax.plot(df['nodes'], df['generation_time_ms'] / 1000, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Generation Time (seconds)')
    ax.set_title('Graph Generation Time Scaling')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Add linear scaling reference
    x_ref = np.array([df['nodes'].min(), df['nodes'].max()])
    y_ref = df['generation_time_ms'].iloc[0] / 1000 * (x_ref / df['nodes'].iloc[0])
    ax.plot(x_ref, y_ref, '--', color='gray', alpha=0.5, label='Linear scaling')
    ax.legend()
    
    # 2. Memory usage vs graph size
    ax = axes[0, 1]
    ax.plot(df['nodes'], df['memory_used_mb'], 's-', linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Memory Used (MB)')
    ax.set_title('Memory Usage for Graph Generation')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 3. Generation rate (nodes per second)
    ax = axes[1, 0]
    gen_rate = df['nodes'] / (df['generation_time_ms'] / 1000)
    ax.bar(range(len(df)), gen_rate, color='green', alpha=0.7)
    ax.set_xlabel('Graph Index')
    ax.set_ylabel('Nodes Generated per Second')
    ax.set_title('Generation Rate')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"{n/1e6:.1f}M" for n in df['nodes']], rotation=45)
    
    # 4. Edges vs nodes relationship
    ax = axes[1, 1]
    ax.scatter(df['nodes'], df['edges'], s=100, alpha=0.7, color='purple')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Number of Edges')
    ax.set_title('Graph Density')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add theoretical lines for different densities
    x_theory = np.logspace(np.log10(df['nodes'].min()), np.log10(df['nodes'].max()), 100)
    ax.plot(x_theory, x_theory, '--', alpha=0.5, label='Linear (sparse)')
    ax.plot(x_theory, x_theory**1.5, '--', alpha=0.5, label='n^1.5')
    ax.plot(x_theory, x_theory**2, '--', alpha=0.5, label='n^2 (dense)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'generation_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_algorithm_performance(results: Dict[str, Any], output_dir: Path):
    """Plot algorithm performance metrics."""
    algo_data = results['algorithm_performance']
    
    if not algo_data:
        print("No algorithm data found")
        return
    
    df = pd.DataFrame(algo_data)
    
    # Group by algorithm
    algorithms = df['algorithm'].unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, algo in enumerate(algorithms[:4]):
        ax = axes[idx]
        algo_df = df[df['algorithm'] == algo]
        
        # Plot execution time vs graph size
        ax.plot(algo_df['graph_size'], algo_df['execution_time_ms'] / 1000, 
                'o-', linewidth=2, markersize=8, label='Actual')
        
        ax.set_xlabel('Graph Size (nodes)')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title(f'{algo} Performance')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add theoretical complexity lines
        x_theory = np.array([algo_df['graph_size'].min(), algo_df['graph_size'].max()])
        
        if 'bfs' in algo.lower() or 'connected' in algo.lower():
            # O(V + E) ~ O(V) for sparse graphs
            y_linear = algo_df['execution_time_ms'].iloc[0] / 1000 * (x_theory / algo_df['graph_size'].iloc[0])
            ax.plot(x_theory, y_linear, '--', alpha=0.5, label='O(V)')
        elif 'pagerank' in algo.lower():
            # O(V + E) per iteration
            y_linear = algo_df['execution_time_ms'].iloc[0] / 1000 * (x_theory / algo_df['graph_size'].iloc[0])
            ax.plot(x_theory, y_linear, '--', alpha=0.5, label='O(V) per iter')
        
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'algorithm_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create algorithm comparison heatmap
    pivot_df = df.pivot_table(
        values='execution_time_ms',
        index='algorithm',
        columns='graph_size',
        aggfunc='mean'
    ) / 1000  # Convert to seconds
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='YlOrRd', 
                cbar_kws={'label': 'Execution Time (seconds)'})
    plt.title('Algorithm Performance Heatmap')
    plt.xlabel('Graph Size (nodes)')
    plt.ylabel('Algorithm')
    plt.tight_layout()
    plt.savefig(output_dir / 'algorithm_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_memory_profile(results: Dict[str, Any], output_dir: Path):
    """Plot memory usage profile."""
    memory_profile = results['memory_profile']
    algo_data = results['algorithm_performance']
    
    if not algo_data:
        return
    
    df = pd.DataFrame(algo_data)
    
    # Memory delta by algorithm
    plt.figure(figsize=(10, 6))
    
    memory_by_algo = df.groupby('algorithm')['memory_delta_mb'].agg(['mean', 'std'])
    memory_by_algo.plot(kind='bar', y='mean', yerr='std', capsize=5, 
                        color='skyblue', edgecolor='black')
    
    plt.xlabel('Algorithm')
    plt.ylabel('Memory Delta (MB)')
    plt.title('Average Memory Usage by Algorithm')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'memory_by_algorithm.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Memory scaling
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo]
        ax.plot(algo_df['graph_size'], algo_df['memory_delta_mb'], 
                'o-', label=algo, markersize=6, alpha=0.7)
    
    ax.set_xlabel('Graph Size (nodes)')
    ax.set_ylabel('Memory Delta (MB)')
    ax.set_title('Memory Usage Scaling by Algorithm')
    ax.set_xscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'memory_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_summary_metrics(results: Dict[str, Any], output_dir: Path):
    """Create summary visualizations."""
    summary = results['summary']
    
    # Create a summary text plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('tight')
    ax.axis('off')
    
    summary_text = f"""
Stress Test Summary
==================

Total Runtime: {summary['total_runtime_seconds']:.2f} seconds
Largest Graph Tested: {summary['largest_graph_tested']:,} nodes

Memory Profile:
- Peak Memory: {results['memory_profile']['peak_memory_mb']:.1f} MB
- Average Memory: {results['memory_profile']['average_memory_mb']:.1f} MB
- Memory Efficiency: {results['memory_profile']['memory_efficiency_ratio']:.2f}

Failures: {len(summary['failures'])}
Warnings: {len(summary['warnings'])}
"""
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=14, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if summary['failures']:
        failures_text = "\nFailures:\n" + "\n".join(f"• {f}" for f in summary['failures'][:5])
        ax.text(0.1, 0.4, failures_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', color='red')
    
    if summary['warnings']:
        warnings_text = "\nWarnings:\n" + "\n".join(f"• {w}" for w in summary['warnings'][:5])
        ax.text(0.1, 0.2, warnings_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', color='orange')
    
    plt.title('Stress Test Summary Report', fontsize=16, fontweight='bold')
    plt.savefig(output_dir / 'summary_report.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_scaling_analysis(results: Dict[str, Any], output_dir: Path):
    """Analyze and visualize scaling behavior."""
    gen_data = pd.DataFrame(results['graph_generation'])
    algo_data = pd.DataFrame(results['algorithm_performance'])
    
    if gen_data.empty or algo_data.empty:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Generation time scaling exponent
    ax = axes[0, 0]
    if len(gen_data) > 2:
        log_nodes = np.log10(gen_data['nodes'])
        log_time = np.log10(gen_data['generation_time_ms'])
        coef = np.polyfit(log_nodes, log_time, 1)
        
        ax.scatter(log_nodes, log_time, s=100, alpha=0.7)
        ax.plot(log_nodes, np.poly1d(coef)(log_nodes), 'r--', 
                label=f'Scaling: O(n^{coef[0]:.2f})')
        
        ax.set_xlabel('log₁₀(Nodes)')
        ax.set_ylabel('log₁₀(Generation Time)')
        ax.set_title('Generation Time Scaling Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Algorithm scaling comparison
    ax = axes[0, 1]
    scaling_exponents = {}
    
    for algo in algo_data['algorithm'].unique():
        algo_df = algo_data[algo_data['algorithm'] == algo]
        if len(algo_df) > 2:
            log_size = np.log10(algo_df['graph_size'])
            log_time = np.log10(algo_df['execution_time_ms'])
            coef = np.polyfit(log_size, log_time, 1)
            scaling_exponents[algo] = coef[0]
    
    if scaling_exponents:
        algos = list(scaling_exponents.keys())
        exponents = list(scaling_exponents.values())
        
        bars = ax.bar(range(len(algos)), exponents, color='green', alpha=0.7)
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Scaling Exponent')
        ax.set_title('Algorithm Scaling Exponents (O(n^x))')
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(algos, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, exp in zip(bars, exponents):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{exp:.2f}', ha='center', va='bottom')
        
        # Add theoretical lines
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Linear O(n)')
        ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Quadratic O(n²)')
        ax.legend()
    
    # 3. Efficiency metric (nodes processed per second per MB)
    ax = axes[1, 0]
    
    gen_efficiency = gen_data['nodes'] / (gen_data['generation_time_ms'] / 1000) / gen_data['memory_used_mb']
    
    ax.plot(gen_data['nodes'], gen_efficiency, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Graph Size (nodes)')
    ax.set_ylabel('Efficiency (nodes/sec/MB)')
    ax.set_title('Memory-Time Efficiency')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # 4. Parallel speedup potential
    ax = axes[1, 1]
    
    # Estimate parallel speedup based on algorithm characteristics
    parallel_potential = {
        'bfs': 0.7,  # Medium parallelization
        'connected_components': 0.8,  # Good parallelization
        'pagerank': 0.9,  # Excellent parallelization
        'degree_distribution': 0.95,  # Near-perfect parallelization
        'clustering_coefficient': 0.85,  # Good parallelization
    }
    
    algos = []
    potentials = []
    for algo, potential in parallel_potential.items():
        if algo in algo_data['algorithm'].values:
            algos.append(algo)
            potentials.append(potential)
    
    if algos:
        bars = ax.bar(range(len(algos)), potentials, color='purple', alpha=0.7)
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Parallel Efficiency Potential')
        ax.set_title('Estimated Parallel Speedup Potential')
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(algos, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        
        # Add percentage labels
        for bar, pot in zip(bars, potentials):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{pot*100:.0f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scaling_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Visualize scirs2-graph stress test results'
    )
    parser.add_argument(
        'results_file',
        help='Path to the JSON results file'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='stress_test_visualizations',
        help='Output directory for visualizations'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load results
    print(f"Loading results from {args.results_file}...")
    results = load_stress_test_results(args.results_file)
    
    # Create visualizations
    print("Creating generation performance plots...")
    plot_generation_performance(results, output_dir)
    
    print("Creating algorithm performance plots...")
    plot_algorithm_performance(results, output_dir)
    
    print("Creating memory profile plots...")
    plot_memory_profile(results, output_dir)
    
    print("Creating summary metrics...")
    plot_summary_metrics(results, output_dir)
    
    print("Creating scaling analysis...")
    create_scaling_analysis(results, output_dir)
    
    print(f"\nVisualization complete! Results saved to {output_dir}/")
    
    # Create index HTML file
    create_html_index(output_dir)

def create_html_index(output_dir: Path):
    """Create an HTML index file for easy viewing of results."""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>scirs2-graph Stress Test Results</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
        }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-top: 20px;
        }
        .image-container {
            background: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        img {
            width: 100%;
            height: auto;
            border-radius: 4px;
        }
        h3 {
            margin-top: 10px;
            color: #555;
        }
    </style>
</head>
<body>
    <h1>scirs2-graph Large Graph Stress Test Results</h1>
    
    <div class="image-grid">
        <div class="image-container">
            <img src="summary_report.png" alt="Summary Report">
            <h3>Summary Report</h3>
        </div>
        
        <div class="image-container">
            <img src="generation_performance.png" alt="Generation Performance">
            <h3>Graph Generation Performance</h3>
        </div>
        
        <div class="image-container">
            <img src="algorithm_performance.png" alt="Algorithm Performance">
            <h3>Algorithm Performance</h3>
        </div>
        
        <div class="image-container">
            <img src="algorithm_heatmap.png" alt="Algorithm Heatmap">
            <h3>Performance Heatmap</h3>
        </div>
        
        <div class="image-container">
            <img src="memory_by_algorithm.png" alt="Memory by Algorithm">
            <h3>Memory Usage by Algorithm</h3>
        </div>
        
        <div class="image-container">
            <img src="memory_scaling.png" alt="Memory Scaling">
            <h3>Memory Scaling Analysis</h3>
        </div>
        
        <div class="image-container">
            <img src="scaling_analysis.png" alt="Scaling Analysis">
            <h3>Comprehensive Scaling Analysis</h3>
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_dir / 'index.html', 'w') as f:
        f.write(html_content)
    
    print(f"HTML index created at {output_dir}/index.html")

if __name__ == "__main__":
    main()