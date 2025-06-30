#!/usr/bin/env python3
"""
Performance Analysis Script for SciRS2 Optimization Library

This script analyzes benchmark results, compares them with baselines,
and generates comprehensive performance reports for CI/CD integration.
"""

import argparse
import json
import os
import glob
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import statistics

# Optional imports with fallback
try:
    import numpy as np
    import scipy.stats as stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: NumPy/SciPy not available. Statistical analysis will be limited.")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: Pandas not available. Data processing will be limited.")


class PerformanceAnalyzer:
    """Main performance analysis engine."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('performance_analysis.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def analyze_benchmark_results(
        self,
        results_dir: str,
        baseline_dir: str,
        features: str,
        commit_hash: str,
        branch: str
    ) -> Dict[str, Any]:
        """Analyze benchmark results and compare with baselines."""
        
        self.logger.info(f"Analyzing performance results for features: {features}")
        
        # Load benchmark results
        benchmark_files = glob.glob(os.path.join(results_dir, "*-benchmarks.json"))
        criterion_files = glob.glob(os.path.join(results_dir, "criterion-results.json"))
        memory_files = glob.glob(os.path.join(results_dir, "memory-benchmarks.json"))
        
        current_results = {}
        
        # Process optimizer benchmarks
        for benchmark_file in benchmark_files:
            self.logger.info(f"Processing benchmark file: {benchmark_file}")
            try:
                with open(benchmark_file, 'r') as f:
                    data = json.load(f)
                    current_results.update(self._process_optimizer_benchmarks(data))
            except Exception as e:
                self.logger.error(f"Error processing {benchmark_file}: {e}")
        
        # Process Criterion results
        for criterion_file in criterion_files:
            self.logger.info(f"Processing Criterion file: {criterion_file}")
            try:
                with open(criterion_file, 'r') as f:
                    data = json.load(f)
                    current_results.update(self._process_criterion_results(data))
            except Exception as e:
                self.logger.error(f"Error processing {criterion_file}: {e}")
        
        # Process memory benchmarks
        for memory_file in memory_files:
            self.logger.info(f"Processing memory file: {memory_file}")
            try:
                with open(memory_file, 'r') as f:
                    data = json.load(f)
                    current_results.update(self._process_memory_benchmarks(data))
            except Exception as e:
                self.logger.error(f"Error processing {memory_file}: {e}")
        
        # Load baseline data
        baseline_data = self._load_baseline_data(baseline_dir, features)
        
        # Perform comparative analysis
        analysis_results = self._perform_comparative_analysis(
            current_results, baseline_data, features, commit_hash, branch
        )
        
        # Generate statistical insights
        if HAS_SCIPY:
            analysis_results['statistical_analysis'] = self._perform_statistical_analysis(
                current_results, baseline_data
            )
        
        # Generate trends analysis
        analysis_results['trend_analysis'] = self._analyze_performance_trends(
            current_results, baseline_data
        )
        
        return analysis_results
    
    def _process_optimizer_benchmarks(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process optimizer benchmark results."""
        processed = {}
        
        for test_name, test_data in data.items():
            if isinstance(test_data, dict) and 'results' in test_data:
                for optimizer_name, optimizer_results in test_data['results'].items():
                    key = f"optimizer_{test_name}_{optimizer_name}"
                    
                    if isinstance(optimizer_results, dict):
                        processed[key] = {
                            'type': 'optimizer_benchmark',
                            'test_name': test_name,
                            'optimizer_name': optimizer_name,
                            'execution_time': optimizer_results.get('execution_time', 0.0),
                            'iterations': optimizer_results.get('iterations', 0),
                            'convergence_rate': optimizer_results.get('convergence_rate', 0.0),
                            'final_error': optimizer_results.get('final_error', float('inf')),
                            'memory_usage': optimizer_results.get('memory_usage', 0),
                            'throughput': optimizer_results.get('throughput', 0.0),
                        }
        
        return processed
    
    def _process_criterion_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Criterion micro-benchmark results."""
        processed = {}
        
        if isinstance(data, dict) and 'benchmarks' in data:
            for benchmark in data['benchmarks']:
                benchmark_name = benchmark.get('name', 'unknown')
                key = f"criterion_{benchmark_name}"
                
                mean_time = 0.0
                std_dev = 0.0
                
                if 'mean' in benchmark and 'estimate' in benchmark['mean']:
                    mean_time = benchmark['mean']['estimate'] / 1e9  # Convert to seconds
                
                if 'std_dev' in benchmark and 'estimate' in benchmark['std_dev']:
                    std_dev = benchmark['std_dev']['estimate'] / 1e9
                
                processed[key] = {
                    'type': 'criterion_benchmark',
                    'benchmark_name': benchmark_name,
                    'mean_time': mean_time,
                    'std_dev': std_dev,
                    'throughput': 1.0 / mean_time if mean_time > 0 else 0.0,
                }
        
        return processed
    
    def _process_memory_benchmarks(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process memory benchmark results."""
        processed = {}
        
        for test_name, test_data in data.items():
            if isinstance(test_data, dict):
                key = f"memory_{test_name}"
                processed[key] = {
                    'type': 'memory_benchmark',
                    'test_name': test_name,
                    'peak_memory': test_data.get('peak_memory', 0),
                    'average_memory': test_data.get('average_memory', 0.0),
                    'memory_efficiency': test_data.get('memory_efficiency', 1.0),
                    'allocations': test_data.get('allocations', 0),
                    'deallocations': test_data.get('deallocations', 0),
                }
        
        return processed
    
    def _load_baseline_data(self, baseline_dir: str, features: str) -> Dict[str, Any]:
        """Load baseline performance data."""
        baseline_file = os.path.join(baseline_dir, f"baseline_{features}.json")
        
        if not os.path.exists(baseline_file):
            self.logger.warning(f"No baseline file found: {baseline_file}")
            return {}
        
        try:
            with open(baseline_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading baseline data: {e}")
            return {}
    
    def _perform_comparative_analysis(
        self,
        current_results: Dict[str, Any],
        baseline_data: Dict[str, Any],
        features: str,
        commit_hash: str,
        branch: str
    ) -> Dict[str, Any]:
        """Perform comparative analysis between current and baseline results."""
        
        analysis = {
            'metadata': {
                'features': features,
                'commit_hash': commit_hash,
                'branch': branch,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'total_benchmarks': len(current_results),
                'baseline_available': bool(baseline_data),
            },
            'current_results': current_results,
            'baseline_comparison': {},
            'performance_summary': {},
            'alerts': [],
        }
        
        if not baseline_data:
            analysis['alerts'].append({
                'type': 'warning',
                'message': 'No baseline data available for comparison',
                'severity': 'medium',
            })
            return analysis
        
        baseline_metrics = baseline_data.get('metrics', {})
        
        # Compare each metric
        for metric_name, current_value in current_results.items():
            if metric_name in baseline_metrics:
                baseline_value = baseline_metrics[metric_name]
                comparison = self._compare_metric(metric_name, current_value, baseline_value)
                analysis['baseline_comparison'][metric_name] = comparison
                
                # Check for significant changes
                if comparison['change_percentage'] > 10:  # 10% threshold
                    analysis['alerts'].append({
                        'type': 'performance_change',
                        'metric': metric_name,
                        'change_percentage': comparison['change_percentage'],
                        'severity': 'high' if comparison['change_percentage'] > 25 else 'medium',
                        'message': f"{metric_name}: {comparison['change_percentage']:.1f}% change from baseline",
                    })
        
        # Generate performance summary
        analysis['performance_summary'] = self._generate_performance_summary(
            current_results, baseline_metrics
        )
        
        return analysis
    
    def _compare_metric(
        self,
        metric_name: str,
        current_value: Any,
        baseline_value: Any
    ) -> Dict[str, Any]:
        """Compare a single metric with its baseline."""
        
        comparison = {
            'metric_name': metric_name,
            'current_value': current_value,
            'baseline_value': baseline_value,
            'change_absolute': 0.0,
            'change_percentage': 0.0,
            'improvement': False,
            'degradation': False,
        }
        
        try:
            # Extract numeric values for comparison
            current_num = self._extract_numeric_value(current_value)
            baseline_num = self._extract_numeric_value(baseline_value)
            
            if baseline_num != 0:
                comparison['change_absolute'] = current_num - baseline_num
                comparison['change_percentage'] = (
                    (current_num - baseline_num) / baseline_num
                ) * 100
                
                # Determine if this is an improvement or degradation
                # Lower is better for: execution_time, memory_usage, error
                # Higher is better for: throughput, convergence_rate
                if 'time' in metric_name.lower() or 'memory' in metric_name.lower() or 'error' in metric_name.lower():
                    comparison['improvement'] = comparison['change_percentage'] < 0
                    comparison['degradation'] = comparison['change_percentage'] > 0
                else:
                    comparison['improvement'] = comparison['change_percentage'] > 0
                    comparison['degradation'] = comparison['change_percentage'] < 0
        
        except (TypeError, ValueError) as e:
            self.logger.warning(f"Could not compare metric {metric_name}: {e}")
        
        return comparison
    
    def _extract_numeric_value(self, value: Any) -> float:
        """Extract numeric value from various data types."""
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, dict):
            # Try common numeric fields
            for field in ['execution_time', 'mean_time', 'peak_memory', 'throughput']:
                if field in value:
                    return float(value[field])
            # Fallback to first numeric value
            for v in value.values():
                if isinstance(v, (int, float)):
                    return float(v)
        return 0.0
    
    def _perform_statistical_analysis(
        self,
        current_results: Dict[str, Any],
        baseline_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform statistical analysis on performance data."""
        if not HAS_SCIPY:
            return {'error': 'SciPy not available for statistical analysis'}
        
        statistical_results = {
            'significance_tests': {},
            'confidence_intervals': {},
            'effect_sizes': {},
        }
        
        baseline_metrics = baseline_data.get('metrics', {})
        
        for metric_name in current_results:
            if metric_name in baseline_metrics:
                try:
                    current_values = self._extract_sample_values(current_results[metric_name])
                    baseline_values = self._extract_sample_values(baseline_metrics[metric_name])
                    
                    if len(current_values) > 1 and len(baseline_values) > 1:
                        # Perform Mann-Whitney U test (non-parametric)
                        statistic, p_value = stats.mannwhitneyu(
                            current_values, baseline_values, alternative='two-sided'
                        )
                        
                        statistical_results['significance_tests'][metric_name] = {
                            'test': 'Mann-Whitney U',
                            'statistic': float(statistic),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05,
                        }
                        
                        # Calculate effect size (Cohen's d)
                        effect_size = self._calculate_cohens_d(current_values, baseline_values)
                        statistical_results['effect_sizes'][metric_name] = effect_size
                        
                        # Calculate confidence interval
                        ci = stats.t.interval(
                            0.95,
                            len(current_values) - 1,
                            loc=np.mean(current_values),
                            scale=stats.sem(current_values)
                        )
                        statistical_results['confidence_intervals'][metric_name] = {
                            'lower': float(ci[0]),
                            'upper': float(ci[1]),
                            'confidence_level': 0.95,
                        }
                
                except Exception as e:
                    self.logger.warning(f"Statistical analysis failed for {metric_name}: {e}")
        
        return statistical_results
    
    def _extract_sample_values(self, data: Any) -> List[float]:
        """Extract sample values for statistical analysis."""
        if isinstance(data, list):
            return [float(x) for x in data if isinstance(x, (int, float))]
        elif isinstance(data, dict):
            # Look for sample data or use single value
            if 'samples' in data:
                return [float(x) for x in data['samples'] if isinstance(x, (int, float))]
            else:
                value = self._extract_numeric_value(data)
                return [value] if value != 0 else []
        elif isinstance(data, (int, float)):
            return [float(data)]
        return []
    
    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        if not group1 or not group2:
            return 0.0
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    def _analyze_performance_trends(
        self,
        current_results: Dict[str, Any],
        baseline_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        
        trend_analysis = {
            'trends': {},
            'predictions': {},
            'anomalies': [],
        }
        
        # Simple trend analysis - would be enhanced with more historical data
        baseline_metrics = baseline_data.get('metrics', {})
        
        for metric_name in current_results:
            if metric_name in baseline_metrics:
                current_val = self._extract_numeric_value(current_results[metric_name])
                baseline_val = self._extract_numeric_value(baseline_metrics[metric_name])
                
                if baseline_val != 0:
                    change_rate = (current_val - baseline_val) / baseline_val
                    
                    trend_analysis['trends'][metric_name] = {
                        'direction': 'improving' if change_rate < 0 else 'degrading' if change_rate > 0 else 'stable',
                        'magnitude': abs(change_rate),
                        'change_rate': change_rate,
                    }
                    
                    # Detect anomalies (significant changes)
                    if abs(change_rate) > 0.2:  # 20% threshold
                        trend_analysis['anomalies'].append({
                            'metric': metric_name,
                            'type': 'significant_change',
                            'change_rate': change_rate,
                            'description': f"{metric_name} changed by {change_rate*100:.1f}%",
                        })
        
        return trend_analysis
    
    def _generate_performance_summary(
        self,
        current_results: Dict[str, Any],
        baseline_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate overall performance summary."""
        
        summary = {
            'total_benchmarks': len(current_results),
            'improved_metrics': 0,
            'degraded_metrics': 0,
            'stable_metrics': 0,
            'overall_score': 0.0,
            'key_metrics': {},
        }
        
        if not baseline_metrics:
            return summary
        
        improvements = []
        degradations = []
        
        for metric_name in current_results:
            if metric_name in baseline_metrics:
                current_val = self._extract_numeric_value(current_results[metric_name])
                baseline_val = self._extract_numeric_value(baseline_metrics[metric_name])
                
                if baseline_val != 0:
                    change_rate = (current_val - baseline_val) / baseline_val
                    
                    # Determine improvement/degradation based on metric type
                    is_lower_better = any(keyword in metric_name.lower() 
                                        for keyword in ['time', 'memory', 'error'])
                    
                    if abs(change_rate) < 0.05:  # 5% threshold for stability
                        summary['stable_metrics'] += 1
                    elif (change_rate < 0 and is_lower_better) or (change_rate > 0 and not is_lower_better):
                        summary['improved_metrics'] += 1
                        improvements.append(abs(change_rate))
                    else:
                        summary['degraded_metrics'] += 1
                        degradations.append(abs(change_rate))
        
        # Calculate overall score
        total_metrics = summary['improved_metrics'] + summary['degraded_metrics'] + summary['stable_metrics']
        if total_metrics > 0:
            improvement_score = summary['improved_metrics'] / total_metrics
            degradation_penalty = summary['degraded_metrics'] / total_metrics
            summary['overall_score'] = max(0.0, improvement_score - degradation_penalty * 0.5)
        
        # Identify key metrics
        if improvements:
            summary['key_metrics']['best_improvement'] = max(improvements)
        if degradations:
            summary['key_metrics']['worst_degradation'] = max(degradations)
        
        return summary


def main():
    """Main entry point for the performance analysis script."""
    parser = argparse.ArgumentParser(description='Analyze performance benchmark results')
    parser.add_argument('--benchmark-results', required=True,
                       help='Directory containing benchmark result files')
    parser.add_argument('--baseline-dir', required=True,
                       help='Directory containing performance baselines')
    parser.add_argument('--output-report', required=True,
                       help='Output file for analysis report')
    parser.add_argument('--features', required=True,
                       help='Feature set being tested')
    parser.add_argument('--commit-hash', required=True,
                       help='Git commit hash')
    parser.add_argument('--branch', required=True,
                       help='Git branch name')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Configuration
    config = {
        'significance_threshold': 0.05,
        'effect_size_threshold': 0.2,
        'change_threshold': 0.1,  # 10%
    }
    
    # Create analyzer and run analysis
    analyzer = PerformanceAnalyzer(config)
    
    try:
        results = analyzer.analyze_benchmark_results(
            args.benchmark_results,
            args.baseline_dir,
            args.features,
            args.commit_hash,
            args.branch
        )
        
        # Save results
        os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
        with open(args.output_report, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Performance analysis completed successfully")
        print(f"📊 Analyzed {results['metadata']['total_benchmarks']} benchmarks")
        
        if results['alerts']:
            print(f"⚠️  {len(results['alerts'])} alerts generated")
            for alert in results['alerts']:
                if alert['severity'] == 'high':
                    print(f"🔴 {alert['message']}")
                elif alert['severity'] == 'medium':
                    print(f"🟡 {alert['message']}")
        
        if results['metadata']['baseline_available']:
            summary = results['performance_summary']
            print(f"📈 Performance Summary:")
            print(f"   Improved: {summary['improved_metrics']}")
            print(f"   Degraded: {summary['degraded_metrics']}")
            print(f"   Stable: {summary['stable_metrics']}")
            print(f"   Overall Score: {summary['overall_score']:.2f}")
        
    except Exception as e:
        logging.error(f"Performance analysis failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()