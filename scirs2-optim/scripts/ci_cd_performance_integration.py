#!/usr/bin/env python3
"""
CI/CD Performance Integration Script for scirs2-optim

This script provides comprehensive CI/CD integration for performance regression testing,
including automated benchmarking, statistical analysis, and reporting for continuous
performance monitoring.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import statistics
import xml.etree.ElementTree as ET

# Third-party imports (would be installed in CI environment)
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class PerformanceMetrics:
    """Container for performance metrics"""
    
    def __init__(self):
        self.execution_time: float = 0.0
        self.memory_usage_mb: float = 0.0
        self.peak_memory_mb: float = 0.0
        self.cpu_utilization: float = 0.0
        self.throughput: float = 0.0
        self.error_rate: float = 0.0
        self.custom_metrics: Dict[str, float] = {}
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'execution_time': self.execution_time,
            'memory_usage_mb': self.memory_usage_mb,
            'peak_memory_mb': self.peak_memory_mb,
            'cpu_utilization': self.cpu_utilization,
            'throughput': self.throughput,
            'error_rate': self.error_rate,
            'custom_metrics': self.custom_metrics
        }


class EnvironmentInfo:
    """System environment information"""
    
    def __init__(self):
        self.os_info = self._get_os_info()
        self.cpu_info = self._get_cpu_info()
        self.memory_info = self._get_memory_info()
        self.rust_version = self._get_rust_version()
        self.compiler_info = self._get_compiler_info()
        self.git_info = self._get_git_info()
        
    def _get_os_info(self) -> Dict[str, str]:
        try:
            import platform
            return {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor()
            }
        except Exception:
            return {'system': 'unknown'}
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        if HAS_PSUTIL:
            return {
                'count': psutil.cpu_count(),
                'count_logical': psutil.cpu_count(logical=True),
                'freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            }
        return {'count': os.cpu_count() or 1}
    
    def _get_memory_info(self) -> Dict[str, int]:
        if HAS_PSUTIL:
            mem = psutil.virtual_memory()
            return {
                'total_mb': mem.total // (1024 * 1024),
                'available_mb': mem.available // (1024 * 1024)
            }
        return {'total_mb': 0}
    
    def _get_rust_version(self) -> str:
        try:
            result = subprocess.run(['rustc', '--version'], capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else 'unknown'
        except Exception:
            return 'unknown'
    
    def _get_compiler_info(self) -> str:
        return self._get_rust_version()
    
    def _get_git_info(self) -> Dict[str, str]:
        try:
            commit_hash = subprocess.run(
                ['git', 'rev-parse', 'HEAD'], 
                capture_output=True, text=True
            ).stdout.strip()
            
            branch = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True, text=True
            ).stdout.strip()
            
            return {
                'commit_hash': commit_hash,
                'branch': branch,
                'short_hash': commit_hash[:8] if commit_hash else 'unknown'
            }
        except Exception:
            return {'commit_hash': 'unknown', 'branch': 'unknown', 'short_hash': 'unknown'}


class BenchmarkRunner:
    """Runs and measures performance benchmarks"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.env_info = EnvironmentInfo()
        
    def run_benchmark_suite(self, config: Dict[str, Any]) -> Dict[str, PerformanceMetrics]:
        """Run all configured benchmarks"""
        results = {}
        
        benchmark_configs = config.get('benchmarks', [])
        for benchmark_config in benchmark_configs:
            benchmark_name = benchmark_config['name']
            print(f"Running benchmark: {benchmark_name}")
            
            try:
                metrics = self._run_single_benchmark(benchmark_config)
                results[benchmark_name] = metrics
                print(f"  ✓ Completed: {benchmark_name}")
            except Exception as e:
                print(f"  ✗ Failed: {benchmark_name} - {e}")
                # Create empty metrics for failed benchmarks
                results[benchmark_name] = PerformanceMetrics()
                results[benchmark_name].error_rate = 1.0
        
        return results
    
    def _run_single_benchmark(self, config: Dict[str, Any]) -> PerformanceMetrics:
        """Run a single benchmark and collect metrics"""
        benchmark_type = config.get('type', 'cargo_bench')
        
        if benchmark_type == 'cargo_bench':
            return self._run_cargo_benchmark(config)
        elif benchmark_type == 'custom_command':
            return self._run_custom_command(config)
        elif benchmark_type == 'criterion':
            return self._run_criterion_benchmark(config)
        else:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}")
    
    def _run_cargo_benchmark(self, config: Dict[str, Any]) -> PerformanceMetrics:
        """Run cargo bench with performance monitoring"""
        cmd = ['cargo', 'bench', '--bench', config['benchmark_name']]
        
        if config.get('features'):
            cmd.extend(['--features', ','.join(config['features'])])
        
        if config.get('release', True):
            cmd.append('--release')
        
        # Add filter if specified
        if config.get('filter'):
            cmd.append(config['filter'])
        
        start_time = time.time()
        initial_memory = self._get_current_memory_usage()
        
        # Run the benchmark
        process = subprocess.Popen(
            cmd,
            cwd=self.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Monitor performance during execution
        peak_memory = initial_memory
        cpu_samples = []
        
        while process.poll() is None:
            if HAS_PSUTIL:
                try:
                    current_memory = self._get_current_memory_usage()
                    peak_memory = max(peak_memory, current_memory)
                    cpu_samples.append(psutil.cpu_percent(interval=0.1))
                except Exception:
                    pass
            time.sleep(0.1)
        
        execution_time = time.time() - start_time
        stdout, stderr = process.communicate()
        
        # Parse benchmark output
        metrics = PerformanceMetrics()
        metrics.execution_time = execution_time
        metrics.memory_usage_mb = peak_memory - initial_memory
        metrics.peak_memory_mb = peak_memory
        metrics.cpu_utilization = statistics.mean(cpu_samples) if cpu_samples else 0.0
        metrics.error_rate = 1.0 if process.returncode != 0 else 0.0
        
        # Try to parse criterion output for more detailed metrics
        if 'time:' in stdout:
            metrics.custom_metrics.update(self._parse_criterion_output(stdout))
        
        return metrics
    
    def _run_custom_command(self, config: Dict[str, Any]) -> PerformanceMetrics:
        """Run a custom command with monitoring"""
        cmd = config['command']
        args = config.get('args', [])
        full_cmd = [cmd] + args
        
        start_time = time.time()
        initial_memory = self._get_current_memory_usage()
        
        process = subprocess.Popen(
            full_cmd,
            cwd=self.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        peak_memory = initial_memory
        cpu_samples = []
        
        while process.poll() is None:
            if HAS_PSUTIL:
                try:
                    current_memory = self._get_current_memory_usage()
                    peak_memory = max(peak_memory, current_memory)
                    cpu_samples.append(psutil.cpu_percent(interval=0.1))
                except Exception:
                    pass
            time.sleep(0.1)
        
        execution_time = time.time() - start_time
        stdout, stderr = process.communicate()
        
        metrics = PerformanceMetrics()
        metrics.execution_time = execution_time
        metrics.memory_usage_mb = peak_memory - initial_memory
        metrics.peak_memory_mb = peak_memory
        metrics.cpu_utilization = statistics.mean(cpu_samples) if cpu_samples else 0.0
        metrics.error_rate = 1.0 if process.returncode != 0 else 0.0
        
        # Try to parse output for throughput information
        if config.get('throughput_pattern'):
            throughput = self._extract_throughput(stdout, config['throughput_pattern'])
            if throughput is not None:
                metrics.throughput = throughput
        
        return metrics
    
    def _run_criterion_benchmark(self, config: Dict[str, Any]) -> PerformanceMetrics:
        """Run criterion benchmarks with detailed analysis"""
        # This would interface with criterion's JSON output
        return self._run_cargo_benchmark(config)
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if HAS_PSUTIL:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        return 0.0
    
    def _parse_criterion_output(self, output: str) -> Dict[str, float]:
        """Parse criterion benchmark output for metrics"""
        metrics = {}
        
        lines = output.split('\n')
        for line in lines:
            if 'time:' in line and 'ns' in line:
                # Extract timing information
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'time:' and i + 2 < len(parts):
                        try:
                            value = float(parts[i + 1])
                            unit = parts[i + 2]
                            
                            # Convert to microseconds
                            if unit == 'ns':
                                value = value / 1000.0
                            elif unit == 'μs' or unit == 'us':
                                pass  # Already in microseconds
                            elif unit == 'ms':
                                value = value * 1000.0
                            elif unit == 's':
                                value = value * 1000000.0
                            
                            metrics['criterion_time_us'] = value
                        except ValueError:
                            pass
        
        return metrics
    
    def _extract_throughput(self, output: str, pattern: str) -> Optional[float]:
        """Extract throughput information from output using a pattern"""
        import re
        
        try:
            match = re.search(pattern, output)
            if match:
                return float(match.group(1))
        except Exception:
            pass
        
        return None


class StatisticalAnalyzer:
    """Performs statistical analysis on performance data"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        
    def analyze_regression(
        self, 
        current_metrics: Dict[str, PerformanceMetrics],
        historical_data: List[Dict[str, PerformanceMetrics]],
        thresholds: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze for performance regressions"""
        
        results = {
            'regressions_detected': [],
            'warnings': [],
            'summary': {},
            'statistical_tests': {}
        }
        
        if not historical_data:
            results['summary']['status'] = 'no_baseline'
            return results
        
        # Get baseline (mean of historical data)
        baseline_metrics = self._calculate_baseline(historical_data)
        
        for benchmark_name, current_metric in current_metrics.items():
            if benchmark_name not in baseline_metrics:
                continue
                
            baseline_metric = baseline_metrics[benchmark_name]
            analysis = self._analyze_single_metric(
                benchmark_name, current_metric, baseline_metric, thresholds
            )
            
            if analysis['is_regression']:
                results['regressions_detected'].append(analysis)
            elif analysis['is_warning']:
                results['warnings'].append(analysis)
            
            results['statistical_tests'][benchmark_name] = analysis['statistical_test']
        
        # Overall status
        if results['regressions_detected']:
            results['summary']['status'] = 'regression_detected'
        elif results['warnings']:
            results['summary']['status'] = 'warning'
        else:
            results['summary']['status'] = 'pass'
        
        results['summary']['total_benchmarks'] = len(current_metrics)
        results['summary']['regressions_count'] = len(results['regressions_detected'])
        results['summary']['warnings_count'] = len(results['warnings'])
        
        return results
    
    def _calculate_baseline(
        self, 
        historical_data: List[Dict[str, PerformanceMetrics]]
    ) -> Dict[str, PerformanceMetrics]:
        """Calculate baseline metrics from historical data"""
        
        baseline = {}
        
        # Get all benchmark names
        all_benchmarks = set()
        for data_point in historical_data:
            all_benchmarks.update(data_point.keys())
        
        for benchmark_name in all_benchmarks:
            # Collect historical values for this benchmark
            execution_times = []
            memory_usages = []
            cpu_utilizations = []
            throughputs = []
            
            for data_point in historical_data:
                if benchmark_name in data_point:
                    metric = data_point[benchmark_name]
                    execution_times.append(metric.execution_time)
                    memory_usages.append(metric.memory_usage_mb)
                    cpu_utilizations.append(metric.cpu_utilization)
                    if metric.throughput > 0:
                        throughputs.append(metric.throughput)
            
            if execution_times:
                baseline_metric = PerformanceMetrics()
                baseline_metric.execution_time = statistics.mean(execution_times)
                baseline_metric.memory_usage_mb = statistics.mean(memory_usages)
                baseline_metric.cpu_utilization = statistics.mean(cpu_utilizations)
                baseline_metric.throughput = statistics.mean(throughputs) if throughputs else 0.0
                
                baseline[benchmark_name] = baseline_metric
        
        return baseline
    
    def _analyze_single_metric(
        self,
        benchmark_name: str,
        current: PerformanceMetrics,
        baseline: PerformanceMetrics,
        thresholds: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze a single metric for regression"""
        
        analysis = {
            'benchmark_name': benchmark_name,
            'is_regression': False,
            'is_warning': False,
            'metrics': {},
            'statistical_test': {}
        }
        
        # Execution time analysis
        if baseline.execution_time > 0:
            time_change = (current.execution_time - baseline.execution_time) / baseline.execution_time
            time_threshold = thresholds.get('execution_time', 0.1)  # 10% default
            
            analysis['metrics']['execution_time'] = {
                'current': current.execution_time,
                'baseline': baseline.execution_time,
                'change_percent': time_change * 100,
                'threshold_percent': time_threshold * 100,
                'is_regression': time_change > time_threshold,
                'is_warning': time_change > time_threshold * 0.5
            }
            
            if time_change > time_threshold:
                analysis['is_regression'] = True
            elif time_change > time_threshold * 0.5:
                analysis['is_warning'] = True
        
        # Memory usage analysis
        if baseline.memory_usage_mb > 0:
            memory_change = (current.memory_usage_mb - baseline.memory_usage_mb) / baseline.memory_usage_mb
            memory_threshold = thresholds.get('memory_usage', 0.2)  # 20% default
            
            analysis['metrics']['memory_usage'] = {
                'current': current.memory_usage_mb,
                'baseline': baseline.memory_usage_mb,
                'change_percent': memory_change * 100,
                'threshold_percent': memory_threshold * 100,
                'is_regression': memory_change > memory_threshold,
                'is_warning': memory_change > memory_threshold * 0.5
            }
            
            if memory_change > memory_threshold:
                analysis['is_regression'] = True
            elif memory_change > memory_threshold * 0.5:
                analysis['is_warning'] = True
        
        # Throughput analysis (lower is worse)
        if baseline.throughput > 0 and current.throughput > 0:
            throughput_change = (current.throughput - baseline.throughput) / baseline.throughput
            throughput_threshold = thresholds.get('throughput', -0.1)  # -10% default
            
            analysis['metrics']['throughput'] = {
                'current': current.throughput,
                'baseline': baseline.throughput,
                'change_percent': throughput_change * 100,
                'threshold_percent': throughput_threshold * 100,
                'is_regression': throughput_change < throughput_threshold,
                'is_warning': throughput_change < throughput_threshold * 0.5
            }
            
            if throughput_change < throughput_threshold:
                analysis['is_regression'] = True
            elif throughput_change < throughput_threshold * 0.5:
                analysis['is_warning'] = True
        
        return analysis


class ReportGenerator:
    """Generates various types of performance reports"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_junit_report(
        self, 
        analysis_results: Dict[str, Any],
        benchmark_results: Dict[str, PerformanceMetrics]
    ) -> Path:
        """Generate JUnit XML report for CI/CD integration"""
        
        root = ET.Element('testsuites')
        root.set('name', 'performance_tests')
        root.set('tests', str(len(benchmark_results)))
        root.set('failures', str(len(analysis_results.get('regressions_detected', []))))
        root.set('time', str(sum(m.execution_time for m in benchmark_results.values())))
        
        suite = ET.SubElement(root, 'testsuite')
        suite.set('name', 'performance_benchmarks')
        suite.set('tests', str(len(benchmark_results)))
        suite.set('failures', str(len(analysis_results.get('regressions_detected', []))))
        
        # Add test cases
        for benchmark_name, metrics in benchmark_results.items():
            testcase = ET.SubElement(suite, 'testcase')
            testcase.set('name', benchmark_name)
            testcase.set('classname', 'performance')
            testcase.set('time', str(metrics.execution_time))
            
            # Check if this benchmark has regressions
            regressions = analysis_results.get('regressions_detected', [])
            benchmark_regressions = [r for r in regressions if r['benchmark_name'] == benchmark_name]
            
            if benchmark_regressions:
                failure = ET.SubElement(testcase, 'failure')
                failure.set('message', f'Performance regression detected in {benchmark_name}')
                failure.text = json.dumps(benchmark_regressions[0], indent=2)
            
            # Add performance data as system-out
            system_out = ET.SubElement(testcase, 'system-out')
            system_out.text = json.dumps(metrics.to_dict(), indent=2)
        
        # Write XML file
        report_path = self.output_dir / 'performance_report.xml'
        tree = ET.ElementTree(root)
        tree.write(report_path, encoding='utf-8', xml_declaration=True)
        
        return report_path
    
    def generate_json_report(
        self,
        analysis_results: Dict[str, Any],
        benchmark_results: Dict[str, PerformanceMetrics],
        env_info: EnvironmentInfo
    ) -> Path:
        """Generate comprehensive JSON report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'environment': {
                'os': env_info.os_info,
                'cpu': env_info.cpu_info,
                'memory': env_info.memory_info,
                'rust_version': env_info.rust_version,
                'git': env_info.git_info
            },
            'benchmark_results': {
                name: metrics.to_dict() 
                for name, metrics in benchmark_results.items()
            },
            'analysis': analysis_results
        }
        
        report_path = self.output_dir / 'performance_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report_path
    
    def generate_markdown_report(
        self,
        analysis_results: Dict[str, Any],
        benchmark_results: Dict[str, PerformanceMetrics],
        env_info: EnvironmentInfo
    ) -> Path:
        """Generate human-readable markdown report"""
        
        lines = []
        lines.append("# Performance Test Report")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Commit: {env_info.git_info.get('commit_hash', 'unknown')}")
        lines.append(f"Branch: {env_info.git_info.get('branch', 'unknown')}")
        lines.append("")
        
        # Summary
        status = analysis_results.get('summary', {}).get('status', 'unknown')
        status_emoji = {
            'pass': '✅',
            'warning': '⚠️',
            'regression_detected': '❌',
            'no_baseline': '📊'
        }.get(status, '❓')
        
        lines.append(f"## Summary {status_emoji}")
        lines.append(f"Status: **{status.replace('_', ' ').title()}**")
        lines.append(f"Total Benchmarks: {len(benchmark_results)}")
        lines.append(f"Regressions: {len(analysis_results.get('regressions_detected', []))}")
        lines.append(f"Warnings: {len(analysis_results.get('warnings', []))}")
        lines.append("")
        
        # Regressions
        regressions = analysis_results.get('regressions_detected', [])
        if regressions:
            lines.append("## ❌ Performance Regressions")
            lines.append("")
            for regression in regressions:
                lines.append(f"### {regression['benchmark_name']}")
                for metric_name, metric_data in regression['metrics'].items():
                    if metric_data.get('is_regression'):
                        lines.append(f"- **{metric_name.replace('_', ' ').title()}**: "
                                   f"{metric_data['change_percent']:.1f}% change "
                                   f"(threshold: {metric_data['threshold_percent']:.1f}%)")
                lines.append("")
        
        # Warnings
        warnings = analysis_results.get('warnings', [])
        if warnings:
            lines.append("## ⚠️ Performance Warnings")
            lines.append("")
            for warning in warnings:
                lines.append(f"### {warning['benchmark_name']}")
                for metric_name, metric_data in warning['metrics'].items():
                    if metric_data.get('is_warning'):
                        lines.append(f"- **{metric_name.replace('_', ' ').title()}**: "
                                   f"{metric_data['change_percent']:.1f}% change")
                lines.append("")
        
        # All results
        lines.append("## 📊 All Benchmark Results")
        lines.append("")
        lines.append("| Benchmark | Execution Time | Memory Usage | CPU Util | Throughput |")
        lines.append("|-----------|----------------|--------------|----------|------------|")
        
        for name, metrics in benchmark_results.items():
            lines.append(f"| {name} | {metrics.execution_time:.3f}s | "
                        f"{metrics.memory_usage_mb:.1f}MB | "
                        f"{metrics.cpu_utilization:.1f}% | "
                        f"{metrics.throughput:.1f} ops/s |")
        
        lines.append("")
        
        # Environment
        lines.append("## 🖥️ Environment")
        lines.append(f"- OS: {env_info.os_info.get('system', 'unknown')}")
        lines.append(f"- CPU: {env_info.cpu_info.get('count', 'unknown')} cores")
        lines.append(f"- Memory: {env_info.memory_info.get('total_mb', 0)}MB")
        lines.append(f"- Rust: {env_info.rust_version}")
        
        report_path = self.output_dir / 'performance_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))
        
        return report_path


class CiCdIntegration:
    """Main CI/CD integration orchestrator"""
    
    def __init__(self, project_root: Path, config_path: Optional[Path] = None):
        self.project_root = project_root
        self.config = self._load_config(config_path)
        self.env_info = EnvironmentInfo()
        
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'benchmarks': [
                {
                    'name': 'optimizer_benchmarks',
                    'type': 'cargo_bench',
                    'benchmark_name': 'optimizer_benchmarks',
                    'features': ['gpu', 'simd'],
                    'release': True
                }
            ],
            'thresholds': {
                'execution_time': 0.1,  # 10%
                'memory_usage': 0.2,    # 20%
                'throughput': -0.1      # -10%
            },
            'historical_data': {
                'enabled': True,
                'file_path': 'performance_history.json',
                'max_entries': 100
            },
            'reports': {
                'output_dir': 'performance_reports',
                'formats': ['json', 'junit', 'markdown']
            },
            'ci_cd': {
                'fail_on_regression': False,
                'exit_code_on_regression': 1,
                'exit_code_on_warning': 0
            }
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    user_config = json.load(f)
                # Merge configurations (user config takes precedence)
                default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
        
        return default_config
    
    def run_performance_tests(self) -> int:
        """Run complete performance test suite and return exit code"""
        
        print("🚀 Starting Performance Test Suite")
        print(f"Project: {self.project_root}")
        print(f"Commit: {self.env_info.git_info.get('short_hash', 'unknown')}")
        print("")
        
        # Run benchmarks
        runner = BenchmarkRunner(self.project_root)
        benchmark_results = runner.run_benchmark_suite(self.config)
        
        # Load historical data
        historical_data = self._load_historical_data()
        
        # Perform analysis
        analyzer = StatisticalAnalyzer()
        analysis_results = analyzer.analyze_regression(
            benchmark_results,
            historical_data,
            self.config['thresholds']
        )
        
        # Save current results to history
        self._save_to_history(benchmark_results)
        
        # Generate reports
        report_generator = ReportGenerator(Path(self.config['reports']['output_dir']))
        
        for report_format in self.config['reports']['formats']:
            if report_format == 'json':
                json_path = report_generator.generate_json_report(
                    analysis_results, benchmark_results, self.env_info
                )
                print(f"📊 JSON report: {json_path}")
            elif report_format == 'junit':
                junit_path = report_generator.generate_junit_report(
                    analysis_results, benchmark_results
                )
                print(f"📋 JUnit report: {junit_path}")
            elif report_format == 'markdown':
                md_path = report_generator.generate_markdown_report(
                    analysis_results, benchmark_results, self.env_info
                )
                print(f"📝 Markdown report: {md_path}")
        
        # Print summary
        self._print_summary(analysis_results)
        
        # Determine exit code
        return self._calculate_exit_code(analysis_results)
    
    def _load_historical_data(self) -> List[Dict[str, PerformanceMetrics]]:
        """Load historical performance data"""
        if not self.config['historical_data']['enabled']:
            return []
        
        history_file = Path(self.config['historical_data']['file_path'])
        if not history_file.exists():
            return []
        
        try:
            with open(history_file) as f:
                raw_data = json.load(f)
            
            # Convert back to PerformanceMetrics objects
            historical_data = []
            for entry in raw_data:
                benchmark_metrics = {}
                for name, metrics_dict in entry.items():
                    metrics = PerformanceMetrics()
                    metrics.execution_time = metrics_dict.get('execution_time', 0.0)
                    metrics.memory_usage_mb = metrics_dict.get('memory_usage_mb', 0.0)
                    metrics.peak_memory_mb = metrics_dict.get('peak_memory_mb', 0.0)
                    metrics.cpu_utilization = metrics_dict.get('cpu_utilization', 0.0)
                    metrics.throughput = metrics_dict.get('throughput', 0.0)
                    metrics.error_rate = metrics_dict.get('error_rate', 0.0)
                    metrics.custom_metrics = metrics_dict.get('custom_metrics', {})
                    benchmark_metrics[name] = metrics
                
                historical_data.append(benchmark_metrics)
            
            return historical_data
            
        except Exception as e:
            print(f"Warning: Could not load historical data: {e}")
            return []
    
    def _save_to_history(self, benchmark_results: Dict[str, PerformanceMetrics]):
        """Save current results to historical data"""
        if not self.config['historical_data']['enabled']:
            return
        
        history_file = Path(self.config['historical_data']['file_path'])
        
        # Load existing data
        historical_data = []
        if history_file.exists():
            try:
                with open(history_file) as f:
                    historical_data = json.load(f)
            except Exception:
                pass
        
        # Add current data
        current_entry = {
            name: metrics.to_dict() 
            for name, metrics in benchmark_results.items()
        }
        historical_data.append(current_entry)
        
        # Trim to max entries
        max_entries = self.config['historical_data']['max_entries']
        if len(historical_data) > max_entries:
            historical_data = historical_data[-max_entries:]
        
        # Save back to file
        try:
            with open(history_file, 'w') as f:
                json.dump(historical_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save historical data: {e}")
    
    def _print_summary(self, analysis_results: Dict[str, Any]):
        """Print performance test summary"""
        summary = analysis_results.get('summary', {})
        status = summary.get('status', 'unknown')
        
        print("\n" + "="*60)
        print("PERFORMANCE TEST SUMMARY")
        print("="*60)
        
        status_messages = {
            'pass': '✅ All benchmarks passed',
            'warning': '⚠️  Performance warnings detected',
            'regression_detected': '❌ Performance regressions detected',
            'no_baseline': '📊 No baseline data available'
        }
        
        print(status_messages.get(status, f'❓ Unknown status: {status}'))
        print(f"Total benchmarks: {summary.get('total_benchmarks', 0)}")
        print(f"Regressions: {summary.get('regressions_count', 0)}")
        print(f"Warnings: {summary.get('warnings_count', 0)}")
        
        # Print detailed regressions
        regressions = analysis_results.get('regressions_detected', [])
        if regressions:
            print("\nDETAILED REGRESSIONS:")
            for regression in regressions:
                print(f"  • {regression['benchmark_name']}")
                for metric_name, metric_data in regression['metrics'].items():
                    if metric_data.get('is_regression'):
                        print(f"    - {metric_name}: {metric_data['change_percent']:.1f}% change")
        
        print("="*60)
    
    def _calculate_exit_code(self, analysis_results: Dict[str, Any]) -> int:
        """Calculate appropriate exit code for CI/CD"""
        summary = analysis_results.get('summary', {})
        status = summary.get('status', 'unknown')
        
        if status == 'regression_detected' and self.config['ci_cd']['fail_on_regression']:
            return self.config['ci_cd']['exit_code_on_regression']
        elif status == 'warning':
            return self.config['ci_cd']['exit_code_on_warning']
        else:
            return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='CI/CD Performance Testing Integration')
    parser.add_argument('--project-root', type=Path, default=Path.cwd(),
                       help='Root directory of the project')
    parser.add_argument('--config', type=Path,
                       help='Configuration file path')
    parser.add_argument('--output-dir', type=Path, default=Path('performance_reports'),
                       help='Output directory for reports')
    parser.add_argument('--fail-on-regression', action='store_true',
                       help='Exit with error code on performance regression')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Override config if command line args provided
    if args.config is None:
        # Look for config in project root
        potential_configs = [
            args.project_root / 'performance_config.json',
            args.project_root / '.performance.json',
            args.project_root / 'ci' / 'performance.json'
        ]
        for config_path in potential_configs:
            if config_path.exists():
                args.config = config_path
                break
    
    integration = CiCdIntegration(args.project_root, args.config)
    
    # Override specific settings from command line
    if args.fail_on_regression:
        integration.config['ci_cd']['fail_on_regression'] = True
    
    integration.config['reports']['output_dir'] = str(args.output_dir)
    
    try:
        exit_code = integration.run_performance_tests()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Performance tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Performance tests failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()