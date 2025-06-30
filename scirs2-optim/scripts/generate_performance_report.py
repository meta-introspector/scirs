#!/usr/bin/env python3
"""
Performance Report Generator for SciRS2 Optimization Library

This script generates comprehensive HTML and Markdown performance reports
from regression analysis results for CI/CD integration and human review.
"""

import argparse
import json
import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
import base64

# Optional imports for enhanced functionality
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: Matplotlib not available. Visual charts will be disabled.")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: Pandas not available. Advanced data processing will be limited.")


class PerformanceReportGenerator:
    """Generate comprehensive performance reports from analysis data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def generate_reports(
        self,
        regression_report_path: str,
        analysis_report_path: str,
        baseline_dir: str,
        output_html: str,
        output_markdown: str,
        features: str
    ) -> None:
        """Generate both HTML and Markdown reports."""
        
        self.logger.info(f"Generating performance reports for features: {features}")
        
        # Load input data
        regression_data = self._load_json_file(regression_report_path)
        analysis_data = self._load_json_file(analysis_report_path)
        baseline_data = self._load_baseline_summary(baseline_dir, features)
        
        # Generate report data structure
        report_data = self._prepare_report_data(
            regression_data, analysis_data, baseline_data, features
        )
        
        # Generate HTML report
        if output_html:
            self._generate_html_report(report_data, output_html)
            self.logger.info(f"HTML report generated: {output_html}")
        
        # Generate Markdown report
        if output_markdown:
            self._generate_markdown_report(report_data, output_markdown)
            self.logger.info(f"Markdown report generated: {output_markdown}")
    
    def _load_json_file(self, file_path: str) -> Dict[str, Any]:
        """Load JSON data from file."""
        if not os.path.exists(file_path):
            self.logger.warning(f"File not found: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return {}
    
    def _load_baseline_summary(self, baseline_dir: str, features: str) -> Dict[str, Any]:
        """Load baseline summary information."""
        baseline_file = os.path.join(baseline_dir, f"baseline_{features}.json")
        return self._load_json_file(baseline_file)
    
    def _prepare_report_data(
        self,
        regression_data: Dict[str, Any],
        analysis_data: Dict[str, Any],
        baseline_data: Dict[str, Any],
        features: str
    ) -> Dict[str, Any]:
        """Prepare consolidated report data."""
        
        now = datetime.now(timezone.utc)
        
        report_data = {
            'metadata': {
                'title': f'Performance Report - {features}',
                'generated_at': now.isoformat(),
                'features': features,
                'commit_hash': analysis_data.get('metadata', {}).get('commit_hash', 'unknown'),
                'branch': analysis_data.get('metadata', {}).get('branch', 'unknown'),
                'total_benchmarks': analysis_data.get('metadata', {}).get('total_benchmarks', 0),
                'baseline_available': analysis_data.get('metadata', {}).get('baseline_available', False),
            },
            'executive_summary': self._generate_executive_summary(regression_data, analysis_data),
            'performance_analysis': analysis_data,
            'regression_analysis': regression_data,
            'baseline_info': baseline_data,
            'detailed_metrics': self._process_detailed_metrics(analysis_data),
            'alerts': analysis_data.get('alerts', []),
            'charts': self._generate_chart_data(analysis_data, regression_data) if HAS_MATPLOTLIB else None,
        }
        
        return report_data
    
    def _generate_executive_summary(
        self,
        regression_data: Dict[str, Any],
        analysis_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate executive summary of performance results."""
        
        summary = analysis_data.get('performance_summary', {})
        alerts = analysis_data.get('alerts', [])
        
        # Count alert severities
        alert_counts = {'high': 0, 'medium': 0, 'low': 0}
        for alert in alerts:
            severity = alert.get('severity', 'medium')
            alert_counts[severity] = alert_counts.get(severity, 0) + 1
        
        # Determine overall status
        overall_status = 'passed'
        if alert_counts['high'] > 0:
            overall_status = 'failed'
        elif alert_counts['medium'] > 3:
            overall_status = 'warning'
        
        regression_status = regression_data.get('status', 'unknown')
        regression_count = regression_data.get('regression_count', 0)
        
        return {
            'overall_status': overall_status,
            'regression_status': regression_status,
            'total_regressions': regression_count,
            'critical_regressions': regression_data.get('critical_regressions', 0),
            'improved_metrics': summary.get('improved_metrics', 0),
            'degraded_metrics': summary.get('degraded_metrics', 0),
            'stable_metrics': summary.get('stable_metrics', 0),
            'overall_score': summary.get('overall_score', 0.0),
            'alert_counts': alert_counts,
            'key_findings': self._extract_key_findings(analysis_data, regression_data),
        }
    
    def _extract_key_findings(
        self,
        analysis_data: Dict[str, Any],
        regression_data: Dict[str, Any]
    ) -> List[str]:
        """Extract key findings from the analysis."""
        findings = []
        
        # Performance improvements
        summary = analysis_data.get('performance_summary', {})
        if summary.get('improved_metrics', 0) > 0:
            findings.append(f"🟢 {summary['improved_metrics']} metrics showed improvement")
        
        # Performance degradations
        if summary.get('degraded_metrics', 0) > 0:
            findings.append(f"🔴 {summary['degraded_metrics']} metrics showed degradation")
        
        # Significant changes
        baseline_comparison = analysis_data.get('baseline_comparison', {})
        significant_changes = [
            metric for metric, data in baseline_comparison.items()
            if abs(data.get('change_percentage', 0)) > 20
        ]
        if significant_changes:
            findings.append(f"⚠️ {len(significant_changes)} metrics had significant changes (>20%)")
        
        # Memory issues
        memory_alerts = [
            alert for alert in analysis_data.get('alerts', [])
            if 'memory' in alert.get('message', '').lower()
        ]
        if memory_alerts:
            findings.append(f"💾 {len(memory_alerts)} memory-related performance issues detected")
        
        # Statistical significance
        if 'statistical_analysis' in analysis_data:
            significant_tests = [
                test for test in analysis_data['statistical_analysis'].get('significance_tests', {}).values()
                if test.get('significant', False)
            ]
            if significant_tests:
                findings.append(f"📊 {len(significant_tests)} statistically significant changes detected")
        
        return findings[:5]  # Limit to top 5 findings
    
    def _process_detailed_metrics(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process detailed metrics for the report."""
        detailed_metrics = []
        
        current_results = analysis_data.get('current_results', {})
        baseline_comparison = analysis_data.get('baseline_comparison', {})
        
        for metric_name, metric_data in current_results.items():
            metric_info = {
                'name': metric_name,
                'type': metric_data.get('type', 'unknown'),
                'current_value': self._format_metric_value(metric_data),
                'baseline_comparison': baseline_comparison.get(metric_name, {}),
                'trend': self._determine_trend(baseline_comparison.get(metric_name, {})),
                'severity': self._determine_severity(baseline_comparison.get(metric_name, {})),
            }
            detailed_metrics.append(metric_info)
        
        # Sort by severity and change magnitude
        detailed_metrics.sort(
            key=lambda x: (
                x['severity'] == 'high',
                x['severity'] == 'medium',
                abs(x['baseline_comparison'].get('change_percentage', 0))
            ),
            reverse=True
        )
        
        return detailed_metrics
    
    def _format_metric_value(self, metric_data: Any) -> str:
        """Format metric value for display."""
        if isinstance(metric_data, dict):
            # Try common numeric fields
            for field in ['execution_time', 'mean_time', 'peak_memory', 'throughput']:
                if field in metric_data:
                    value = metric_data[field]
                    if field in ['execution_time', 'mean_time']:
                        return f"{value:.4f}s"
                    elif field == 'peak_memory':
                        return f"{value / 1024 / 1024:.2f}MB"
                    elif field == 'throughput':
                        return f"{value:.2f} ops/s"
                    else:
                        return f"{value:.4f}"
            return str(metric_data)
        elif isinstance(metric_data, (int, float)):
            return f"{metric_data:.4f}"
        else:
            return str(metric_data)
    
    def _determine_trend(self, comparison_data: Dict[str, Any]) -> str:
        """Determine performance trend."""
        if not comparison_data:
            return 'unknown'
        
        change_pct = comparison_data.get('change_percentage', 0)
        is_improvement = comparison_data.get('improvement', False)
        is_degradation = comparison_data.get('degradation', False)
        
        if abs(change_pct) < 5:
            return 'stable'
        elif is_improvement:
            return 'improving'
        elif is_degradation:
            return 'degrading'
        else:
            return 'changed'
    
    def _determine_severity(self, comparison_data: Dict[str, Any]) -> str:
        """Determine severity level of metric change."""
        if not comparison_data:
            return 'low'
        
        change_pct = abs(comparison_data.get('change_percentage', 0))
        is_degradation = comparison_data.get('degradation', False)
        
        if is_degradation and change_pct > 25:
            return 'high'
        elif is_degradation and change_pct > 10:
            return 'medium'
        elif change_pct > 50:
            return 'medium'
        else:
            return 'low'
    
    def _generate_chart_data(
        self,
        analysis_data: Dict[str, Any],
        regression_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate charts and return as base64 encoded images."""
        if not HAS_MATPLOTLIB:
            return {}
        
        charts = {}
        
        try:
            # Performance summary pie chart
            summary = analysis_data.get('performance_summary', {})
            if summary:
                charts['performance_summary'] = self._create_performance_summary_chart(summary)
            
            # Metric changes bar chart
            baseline_comparison = analysis_data.get('baseline_comparison', {})
            if baseline_comparison:
                charts['metric_changes'] = self._create_metric_changes_chart(baseline_comparison)
            
        except Exception as e:
            self.logger.warning(f"Error generating charts: {e}")
        
        return charts
    
    def _create_performance_summary_chart(self, summary: Dict[str, Any]) -> str:
        """Create performance summary pie chart."""
        labels = []
        sizes = []
        colors = []
        
        if summary.get('improved_metrics', 0) > 0:
            labels.append('Improved')
            sizes.append(summary['improved_metrics'])
            colors.append('#4CAF50')
        
        if summary.get('stable_metrics', 0) > 0:
            labels.append('Stable')
            sizes.append(summary['stable_metrics'])
            colors.append('#2196F3')
        
        if summary.get('degraded_metrics', 0) > 0:
            labels.append('Degraded')
            sizes.append(summary['degraded_metrics'])
            colors.append('#FF5722')
        
        if not labels:
            return ""
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Performance Metrics Distribution')
        
        return self._fig_to_base64(fig)
    
    def _create_metric_changes_chart(self, baseline_comparison: Dict[str, Any]) -> str:
        """Create metric changes bar chart."""
        metrics = list(baseline_comparison.keys())[:10]  # Top 10 metrics
        changes = [baseline_comparison[m].get('change_percentage', 0) for m in metrics]
        
        if not metrics:
            return ""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['green' if c < 0 else 'red' if c > 0 else 'blue' for c in changes]
        
        bars = ax.bar(range(len(metrics)), changes, color=colors, alpha=0.7)
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Change Percentage (%)')
        ax.set_title('Performance Changes from Baseline')
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, change in zip(bars, changes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                   f'{change:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        import io
        
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        
        return image_base64
    
    def _generate_html_report(self, report_data: Dict[str, Any], output_file: str) -> None:
        """Generate comprehensive HTML report."""
        
        html_content = self._build_html_content(report_data)
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _build_html_content(self, report_data: Dict[str, Any]) -> str:
        """Build HTML content for the report."""
        
        metadata = report_data['metadata']
        summary = report_data['executive_summary']
        charts = report_data.get('charts', {})
        
        # Generate charts HTML
        charts_html = ""
        if charts:
            if 'performance_summary' in charts and charts['performance_summary']:
                charts_html += f'''
                <div class="chart-container">
                    <h3>Performance Summary</h3>
                    <img src="data:image/png;base64,{charts['performance_summary']}" alt="Performance Summary Chart" class="chart-image">
                </div>
                '''
            
            if 'metric_changes' in charts and charts['metric_changes']:
                charts_html += f'''
                <div class="chart-container">
                    <h3>Metric Changes from Baseline</h3>
                    <img src="data:image/png;base64,{charts['metric_changes']}" alt="Metric Changes Chart" class="chart-image">
                </div>
                '''
        
        # Generate detailed metrics table
        metrics_html = self._generate_metrics_table_html(report_data['detailed_metrics'])
        
        # Generate alerts HTML
        alerts_html = self._generate_alerts_html(report_data['alerts'])
        
        # Status indicator
        status_color = {
            'passed': '#4CAF50',
            'warning': '#FF9800',
            'failed': '#F44336'
        }.get(summary['overall_status'], '#757575')
        
        status_icon = {
            'passed': '✅',
            'warning': '⚠️',
            'failed': '❌'
        }.get(summary['overall_status'], '❓')
        
        html_template = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{metadata['title']}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #e0e0e0;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .status-badge {{
            display: inline-block;
            padding: 10px 20px;
            border-radius: 25px;
            color: white;
            font-weight: bold;
            font-size: 18px;
            background-color: {status_color};
            margin: 10px 0;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .summary-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
            text-align: center;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .summary-value {{
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }}
        .chart-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .chart-image {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .metrics-table th, .metrics-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .metrics-table th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        .trend-improving {{ color: #4CAF50; }}
        .trend-degrading {{ color: #F44336; }}
        .trend-stable {{ color: #2196F3; }}
        .severity-high {{ background-color: #ffebee; }}
        .severity-medium {{ background-color: #fff3e0; }}
        .severity-low {{ background-color: #e8f5e8; }}
        .alert {{
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid;
        }}
        .alert-high {{
            background-color: #ffebee;
            border-left-color: #f44336;
        }}
        .alert-medium {{
            background-color: #fff3e0;
            border-left-color: #ff9800;
        }}
        .alert-low {{
            background-color: #e8f5e8;
            border-left-color: #4caf50;
        }}
        .metadata {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            font-size: 14px;
        }}
        .key-findings {{
            background: #e3f2fd;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .key-findings ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{metadata['title']}</h1>
            <div class="status-badge">
                {status_icon} {summary['overall_status'].upper()}
            </div>
            <div class="metadata">
                <strong>Generated:</strong> {metadata['generated_at']} | 
                <strong>Commit:</strong> {metadata['commit_hash'][:8]} | 
                <strong>Branch:</strong> {metadata['branch']} | 
                <strong>Features:</strong> {metadata['features']}
            </div>
        </div>

        <div class="summary-grid">
            <div class="summary-card">
                <h3>Total Benchmarks</h3>
                <div class="summary-value">{metadata['total_benchmarks']}</div>
            </div>
            <div class="summary-card">
                <h3>Regressions</h3>
                <div class="summary-value">{summary['total_regressions']}</div>
            </div>
            <div class="summary-card">
                <h3>Improved</h3>
                <div class="summary-value">{summary['improved_metrics']}</div>
            </div>
            <div class="summary-card">
                <h3>Degraded</h3>
                <div class="summary-value">{summary['degraded_metrics']}</div>
            </div>
            <div class="summary-card">
                <h3>Overall Score</h3>
                <div class="summary-value">{summary['overall_score']:.2f}</div>
            </div>
        </div>

        <div class="key-findings">
            <h2>🔍 Key Findings</h2>
            <ul>
                {''.join(f'<li>{finding}</li>' for finding in summary['key_findings'])}
            </ul>
        </div>

        {charts_html}

        <h2>📊 Detailed Metrics</h2>
        {metrics_html}

        <h2>🚨 Alerts</h2>
        {alerts_html}

        <div class="metadata">
            <p><strong>Report Generation Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            <p><strong>Baseline Available:</strong> {'Yes' if metadata['baseline_available'] else 'No'}</p>
        </div>
    </div>
</body>
</html>
        '''
        
        return html_template
    
    def _generate_metrics_table_html(self, metrics: List[Dict[str, Any]]) -> str:
        """Generate HTML table for detailed metrics."""
        if not metrics:
            return "<p>No metrics data available.</p>"
        
        rows_html = ""
        for metric in metrics:
            comparison = metric['baseline_comparison']
            change_pct = comparison.get('change_percentage', 0)
            trend_class = f"trend-{metric['trend']}"
            severity_class = f"severity-{metric['severity']}"
            
            change_display = f"{change_pct:+.1f}%" if comparison else "N/A"
            
            rows_html += f'''
            <tr class="{severity_class}">
                <td>{metric['name'].replace('_', ' ').title()}</td>
                <td>{metric['type']}</td>
                <td>{metric['current_value']}</td>
                <td class="{trend_class}">{change_display}</td>
                <td class="{trend_class}">{metric['trend'].title()}</td>
                <td>{metric['severity'].title()}</td>
            </tr>
            '''
        
        return f'''
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Type</th>
                    <th>Current Value</th>
                    <th>Change from Baseline</th>
                    <th>Trend</th>
                    <th>Severity</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        '''
    
    def _generate_alerts_html(self, alerts: List[Dict[str, Any]]) -> str:
        """Generate HTML for alerts section."""
        if not alerts:
            return "<p>✅ No alerts generated. All metrics are within acceptable ranges.</p>"
        
        alerts_html = ""
        for alert in alerts:
            severity = alert.get('severity', 'medium')
            alert_class = f"alert alert-{severity}"
            icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(severity, '🔵')
            
            alerts_html += f'''
            <div class="{alert_class}">
                <strong>{icon} {alert.get('type', 'Alert').replace('_', ' ').title()}</strong>
                <br>
                {alert.get('message', 'No message available')}
            </div>
            '''
        
        return alerts_html
    
    def _generate_markdown_report(self, report_data: Dict[str, Any], output_file: str) -> None:
        """Generate Markdown report for CI integration."""
        
        metadata = report_data['metadata']
        summary = report_data['executive_summary']
        
        # Status emoji
        status_emoji = {
            'passed': '✅',
            'warning': '⚠️',
            'failed': '❌'
        }.get(summary['overall_status'], '❓')
        
        markdown_content = f"""# {metadata['title']}

{status_emoji} **Status:** {summary['overall_status'].upper()}

## Summary

| Metric | Value |
|--------|-------|
| Total Benchmarks | {metadata['total_benchmarks']} |
| Regressions Detected | {summary['total_regressions']} |
| Critical Regressions | {summary['critical_regressions']} |
| Improved Metrics | {summary['improved_metrics']} |
| Degraded Metrics | {summary['degraded_metrics']} |
| Stable Metrics | {summary['stable_metrics']} |
| Overall Score | {summary['overall_score']:.2f} |

## Key Findings

{chr(10).join(f'- {finding}' for finding in summary['key_findings'])}

## Performance Changes

"""
        
        # Add detailed metrics table
        if report_data['detailed_metrics']:
            markdown_content += "| Metric | Current Value | Change | Trend | Severity |\n"
            markdown_content += "|--------|---------------|--------|-------|----------|\n"
            
            for metric in report_data['detailed_metrics'][:10]:  # Top 10
                comparison = metric['baseline_comparison']
                change_pct = comparison.get('change_percentage', 0)
                change_display = f"{change_pct:+.1f}%" if comparison else "N/A"
                
                trend_emoji = {
                    'improving': '📈',
                    'degrading': '📉',
                    'stable': '➡️',
                    'changed': '🔄',
                    'unknown': '❓'
                }.get(metric['trend'], '❓')
                
                severity_emoji = {
                    'high': '🔴',
                    'medium': '🟡',
                    'low': '🟢'
                }.get(metric['severity'], '🔵')
                
                markdown_content += f"| {metric['name'].replace('_', ' ').title()} | {metric['current_value']} | {change_display} | {trend_emoji} {metric['trend'].title()} | {severity_emoji} {metric['severity'].title()} |\n"
        
        # Add alerts section
        if report_data['alerts']:
            markdown_content += f"\n## Alerts ({len(report_data['alerts'])})\n\n"
            
            for alert in report_data['alerts']:
                severity = alert.get('severity', 'medium')
                icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(severity, '🔵')
                markdown_content += f"- {icon} **{alert.get('type', 'Alert').replace('_', ' ').title()}:** {alert.get('message', 'No message')}\n"
        else:
            markdown_content += "\n## Alerts\n\n✅ No alerts generated.\n"
        
        # Add metadata
        markdown_content += f"""
---

**Generated:** {metadata['generated_at']}  
**Commit:** {metadata['commit_hash']}  
**Branch:** {metadata['branch']}  
**Features:** {metadata['features']}  
**Baseline Available:** {'Yes' if metadata['baseline_available'] else 'No'}
"""
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)


def main():
    """Main entry point for the performance report generator."""
    parser = argparse.ArgumentParser(description='Generate performance reports from analysis data')
    parser.add_argument('--regression-report', required=True,
                       help='Path to regression analysis report JSON file')
    parser.add_argument('--analysis-report', required=True,
                       help='Path to performance analysis report JSON file')
    parser.add_argument('--baseline-dir', required=True,
                       help='Directory containing performance baselines')
    parser.add_argument('--output-html', required=False,
                       help='Output path for HTML report')
    parser.add_argument('--output-markdown', required=False,
                       help='Output path for Markdown report')
    parser.add_argument('--features', required=True,
                       help='Feature set being tested')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.output_html and not args.output_markdown:
        print("Error: At least one output format (--output-html or --output-markdown) must be specified")
        sys.exit(1)
    
    # Configuration
    config = {
        'include_charts': HAS_MATPLOTLIB,
        'max_metrics_display': 20,
        'chart_dpi': 150,
    }
    
    # Create report generator and generate reports
    generator = PerformanceReportGenerator(config)
    
    try:
        generator.generate_reports(
            args.regression_report,
            args.analysis_report,
            args.baseline_dir,
            args.output_html,
            args.output_markdown,
            args.features
        )
        
        print("✅ Performance reports generated successfully")
        if args.output_html:
            print(f"📄 HTML report: {args.output_html}")
        if args.output_markdown:
            print(f"📝 Markdown report: {args.output_markdown}")
        
    except Exception as e:
        logging.error(f"Report generation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()