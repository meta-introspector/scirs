#!/usr/bin/env python3
"""
Performance Monitoring Script for scirs2-special

This script provides comprehensive performance monitoring and regression detection
for the special functions library. It can be used in CI/CD pipelines or for
local performance testing.

Features:
- Benchmark execution and result collection
- Historical performance tracking
- Regression detection with configurable thresholds
- Performance visualization and reporting
- Integration with various CI/CD systems

Usage:
    python performance_monitor.py [options]
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sqlite3
import statistics

# Try to import optional dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class PerformanceMonitor:
    """Main performance monitoring class."""
    
    def __init__(self, db_path: str = "performance.db"):
        self.db_path = Path(db_path)
        self.init_database()
        
        # Performance thresholds
        self.regression_threshold = 1.15  # 15% slowdown = regression
        self.warning_threshold = 1.10     # 10% slowdown = warning
        self.improvement_threshold = 0.90  # 10% speedup = improvement
        
    def init_database(self):
        """Initialize SQLite database for storing performance data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                commit_hash TEXT,
                branch_name TEXT,
                benchmark_name TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT NOT NULL,
                std_dev REAL,
                min_value REAL,
                max_value REAL,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_baselines (
                benchmark_name TEXT PRIMARY KEY,
                baseline_value REAL NOT NULL,
                baseline_timestamp TEXT NOT NULL,
                last_updated TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_benchmark_timestamp 
            ON benchmark_results(benchmark_name, timestamp)
        ''')
        
        conn.commit()
        conn.close()
        
    def run_benchmarks(self) -> Dict[str, Dict]:
        """Execute Criterion benchmarks and collect results."""
        print("🚀 Running performance benchmarks...")
        
        # Run Criterion benchmarks
        cmd = [
            "cargo", "criterion",
            "--bench", "performance_regression",
            "--message-format", "json"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                print(f"❌ Benchmark execution failed: {result.stderr}")
                return {}
                
        except subprocess.TimeoutExpired:
            print("❌ Benchmark execution timed out")
            return {}
        except subprocess.CalledProcessError as e:
            print(f"❌ Benchmark execution error: {e}")
            return {}
            
        # Parse Criterion JSON output
        benchmarks = {}
        for line in result.stdout.split('\n'):
            if line.strip():
                try:
                    data = json.loads(line)
                    if data.get('reason') == 'benchmark-complete':
                        bench_id = data.get('id', '')
                        typical = data.get('typical', {})
                        
                        benchmarks[bench_id] = {
                            'estimate': typical.get('estimate', 0),
                            'unit': typical.get('unit', 'ns'),
                            'lower_bound': typical.get('lower_bound', 0),
                            'upper_bound': typical.get('upper_bound', 0),
                            'std_dev': typical.get('std_dev', 0),
                        }
                except json.JSONDecodeError:
                    continue
                    
        print(f"✅ Collected {len(benchmarks)} benchmark results")
        return benchmarks
        
    def store_results(self, benchmarks: Dict[str, Dict], 
                     commit_hash: Optional[str] = None,
                     branch_name: Optional[str] = None):
        """Store benchmark results in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        for bench_name, data in benchmarks.items():
            cursor.execute('''
                INSERT INTO benchmark_results 
                (timestamp, commit_hash, branch_name, benchmark_name, 
                 value, unit, std_dev, min_value, max_value)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp, commit_hash, branch_name, bench_name,
                data['estimate'], data['unit'], data.get('std_dev'),
                data.get('lower_bound'), data.get('upper_bound')
            ))
            
        conn.commit()
        conn.close()
        
        print(f"📊 Stored {len(benchmarks)} results in database")
        
    def get_baseline_values(self) -> Dict[str, float]:
        """Get baseline performance values for comparison."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT benchmark_name, baseline_value FROM performance_baselines')
        baselines = dict(cursor.fetchall())
        
        conn.close()
        return baselines
        
    def update_baselines(self, benchmarks: Dict[str, Dict]):
        """Update baseline values with current results."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        for bench_name, data in benchmarks.items():
            cursor.execute('''
                INSERT OR REPLACE INTO performance_baselines
                (benchmark_name, baseline_value, baseline_timestamp, last_updated)
                VALUES (?, ?, ?, ?)
            ''', (bench_name, data['estimate'], timestamp, timestamp))
            
        conn.commit()
        conn.close()
        
        print(f"🎯 Updated baselines for {len(benchmarks)} benchmarks")
        
    def detect_regressions(self, benchmarks: Dict[str, Dict]) -> Tuple[List, List, List]:
        """Detect performance regressions by comparing with baselines."""
        baselines = self.get_baseline_values()
        
        regressions = []
        warnings = []
        improvements = []
        
        for bench_name, data in benchmarks.items():
            if bench_name not in baselines:
                continue
                
            current_time = data['estimate']
            baseline_time = baselines[bench_name]
            
            if baseline_time <= 0:
                continue
                
            ratio = current_time / baseline_time
            
            regression_info = {
                'name': bench_name,
                'ratio': ratio,
                'current': current_time,
                'baseline': baseline_time,
                'unit': data['unit']
            }
            
            if ratio >= self.regression_threshold:
                regressions.append(regression_info)
            elif ratio >= self.warning_threshold:
                warnings.append(regression_info)
            elif ratio <= self.improvement_threshold:
                improvements.append(regression_info)
                
        return regressions, warnings, improvements
        
    def generate_report(self, benchmarks: Dict[str, Dict]) -> str:
        """Generate a comprehensive performance report."""
        regressions, warnings, improvements = self.detect_regressions(benchmarks)
        
        report = f"""# Performance Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total benchmarks: {len(benchmarks)}
- Regressions: {len(regressions)}
- Warnings: {len(warnings)}
- Improvements: {len(improvements)}

## Detailed Results
"""
        
        # Add regression details
        if regressions:
            report += "\n### 🚨 Performance Regressions\n\n"
            report += "| Benchmark | Current | Baseline | Slowdown | Unit |\n"
            report += "|-----------|---------|----------|----------|------|\n"
            
            for reg in regressions:
                report += f"| {reg['name']} | {reg['current']:.2f} | {reg['baseline']:.2f} | {reg['ratio']:.2f}x | {reg['unit']} |\n"
                
        # Add warning details
        if warnings:
            report += "\n### ⚠️ Performance Warnings\n\n"
            report += "| Benchmark | Current | Baseline | Slowdown | Unit |\n"
            report += "|-----------|---------|----------|----------|------|\n"
            
            for warn in warnings:
                report += f"| {warn['name']} | {warn['current']:.2f} | {warn['baseline']:.2f} | {warn['ratio']:.2f}x | {warn['unit']} |\n"
                
        # Add improvement details
        if improvements:
            report += "\n### ✅ Performance Improvements\n\n"
            report += "| Benchmark | Current | Baseline | Speedup | Unit |\n"
            report += "|-----------|---------|----------|---------|------|\n"
            
            for imp in improvements:
                report += f"| {imp['name']} | {imp['current']:.2f} | {imp['baseline']:.2f} | {1/imp['ratio']:.2f}x | {imp['unit']} |\n"
                
        # Add all benchmark results
        report += "\n### 📈 All Benchmark Results\n\n"
        report += "| Benchmark | Time | Unit | Std Dev |\n"
        report += "|-----------|------|------|----------|\n"
        
        for name, data in sorted(benchmarks.items()):
            std_dev = data.get('std_dev', 0)
            report += f"| {name} | {data['estimate']:.2f} | {data['unit']} | {std_dev:.2f} |\n"
            
        return report
        
    def plot_trends(self, output_path: str = "performance_trends.png"):
        """Generate performance trend plots."""
        if not HAS_MATPLOTLIB:
            print("⚠️ Matplotlib not available, skipping trend plots")
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get data for the last 30 days
        thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
        
        cursor.execute('''
            SELECT benchmark_name, timestamp, value
            FROM benchmark_results
            WHERE timestamp > ?
            ORDER BY benchmark_name, timestamp
        ''', (thirty_days_ago,))
        
        data = cursor.fetchall()
        conn.close()
        
        if not data:
            print("⚠️ No historical data available for trend analysis")
            return
            
        # Group data by benchmark
        benchmark_data = {}
        for bench_name, timestamp, value in data:
            if bench_name not in benchmark_data:
                benchmark_data[bench_name] = {'timestamps': [], 'values': []}
            benchmark_data[bench_name]['timestamps'].append(datetime.fromisoformat(timestamp))
            benchmark_data[bench_name]['values'].append(value)
            
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot top 4 most important benchmarks
        important_benchmarks = [
            'core_single_value/gamma_medium',
            'core_single_value/erf_medium', 
            'core_single_value/bessel_j0_medium',
            'array_operations/gamma_array_scalar/1000'
        ]
        
        for i, bench_name in enumerate(important_benchmarks):
            if i >= 4 or bench_name not in benchmark_data:
                continue
                
            ax = axes[i]
            data_points = benchmark_data[bench_name]
            
            ax.plot(data_points['timestamps'], data_points['values'], 'o-', linewidth=2, markersize=4)
            ax.set_title(bench_name.replace('/', ' / '), fontsize=10)
            ax.set_ylabel('Time (ns)')
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Trend plots saved to {output_path}")
        
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Remove old benchmark data to keep database size manageable."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
        
        cursor.execute('DELETE FROM benchmark_results WHERE timestamp < ?', (cutoff_date,))
        rows_deleted = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        if rows_deleted > 0:
            print(f"🧹 Cleaned up {rows_deleted} old benchmark records")
            
    def export_data(self, output_path: str, format: str = 'json'):
        """Export benchmark data to various formats."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM benchmark_results 
            ORDER BY timestamp DESC 
            LIMIT 1000
        ''')
        
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        
        if format.lower() == 'json':
            data = [dict(zip(columns, row)) for row in rows]
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif format.lower() == 'csv':
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(columns)
                writer.writerows(rows)
                
        print(f"📤 Exported {len(rows)} records to {output_path}")


def get_git_info() -> Tuple[Optional[str], Optional[str]]:
    """Get current git commit hash and branch name."""
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        branch_name = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        return commit_hash, branch_name
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


def main():
    parser = argparse.ArgumentParser(description='Performance monitoring for scirs2-special')
    parser.add_argument('--db-path', default='performance.db', 
                       help='Path to SQLite database for storing results')
    parser.add_argument('--run-benchmarks', action='store_true',
                       help='Run benchmark suite')
    parser.add_argument('--update-baselines', action='store_true',
                       help='Update baseline values with current results')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate performance report')
    parser.add_argument('--plot-trends', action='store_true',
                       help='Generate trend plots')
    parser.add_argument('--export-data', 
                       help='Export data to file (specify path)')
    parser.add_argument('--export-format', choices=['json', 'csv'], default='json',
                       help='Export format')
    parser.add_argument('--cleanup-days', type=int, default=90,
                       help='Number of days of data to keep')
    parser.add_argument('--regression-threshold', type=float, default=1.15,
                       help='Regression threshold (default: 1.15 = 15% slowdown)')
    parser.add_argument('--warning-threshold', type=float, default=1.10,
                       help='Warning threshold (default: 1.10 = 10% slowdown)')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = PerformanceMonitor(args.db_path)
    monitor.regression_threshold = args.regression_threshold
    monitor.warning_threshold = args.warning_threshold
    
    # Get git information
    commit_hash, branch_name = get_git_info()
    
    benchmarks = {}
    
    # Run benchmarks if requested
    if args.run_benchmarks:
        benchmarks = monitor.run_benchmarks()
        if benchmarks:
            monitor.store_results(benchmarks, commit_hash, branch_name)
    
    # Update baselines if requested
    if args.update_baselines:
        if not benchmarks:
            print("⚠️ No benchmark results available to update baselines")
        else:
            monitor.update_baselines(benchmarks)
    
    # Generate report
    if args.generate_report:
        if not benchmarks:
            print("⚠️ No benchmark results available for report generation")
        else:
            report = monitor.generate_report(benchmarks)
            
            # Write report to file
            with open('performance_report.md', 'w') as f:
                f.write(report)
            print("📊 Performance report written to performance_report.md")
            
            # Check for regressions and set exit code
            regressions, warnings, improvements = monitor.detect_regressions(benchmarks)
            
            if regressions:
                print(f"\n❌ PERFORMANCE REGRESSIONS DETECTED: {len(regressions)}")
                for reg in regressions:
                    print(f"  - {reg['name']}: {reg['ratio']:.2f}x slower")
                sys.exit(1)
            elif warnings:
                print(f"\n⚠️ PERFORMANCE WARNINGS: {len(warnings)}")
                for warn in warnings:
                    print(f"  - {warn['name']}: {warn['ratio']:.2f}x slower")
            else:
                print("\n✅ No performance regressions detected")
    
    # Generate trend plots
    if args.plot_trends:
        monitor.plot_trends()
    
    # Export data
    if args.export_data:
        monitor.export_data(args.export_data, args.export_format)
    
    # Cleanup old data
    monitor.cleanup_old_data(args.cleanup_days)


if __name__ == '__main__':
    main()