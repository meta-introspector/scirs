#!/usr/bin/env python3
"""
Performance Dashboard for scirs2-special

This script provides a comprehensive performance monitoring dashboard that can:
- Parse benchmark results from Criterion.rs
- Compare against historical baselines
- Generate interactive visualizations
- Detect performance regressions and improvements
- Export reports in multiple formats

Usage:
    python3 performance_dashboard.py --help
    python3 performance_dashboard.py --benchmark-dir ./target/criterion
    python3 performance_dashboard.py --interactive --port 8050
    python3 performance_dashboard.py --export-html performance_report.html
"""

import argparse
import json
import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import subprocess
import tempfile

# Optional imports for interactive dashboard
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import dash
    from dash import dcc, html, Input, Output, State
    INTERACTIVE_AVAILABLE = True
except ImportError:
    INTERACTIVE_AVAILABLE = False
    print("Warning: Interactive dashboard not available. Install plotly and dash for full functionality.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceDatabase:
    """SQLite database for storing performance metrics over time"""
    
    def __init__(self, db_path: str = "performance_history.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the performance database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for storing performance data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                commit_sha TEXT,
                branch TEXT,
                rust_version TEXT,
                features TEXT,
                environment_info TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                test_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT,
                std_dev REAL,
                sample_count INTEGER,
                FOREIGN KEY (run_id) REFERENCES performance_runs (id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS regression_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                test_name TEXT NOT NULL,
                severity TEXT NOT NULL,
                change_percent REAL NOT NULL,
                p_value REAL,
                created_at TEXT NOT NULL,
                resolved_at TEXT,
                FOREIGN KEY (run_id) REFERENCES performance_runs (id)
            )
        """)
        
        # Create indexes for better query performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_benchmark_test_name ON benchmark_results (test_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_runs (timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_regression_severity ON regression_alerts (severity)")
        
        conn.commit()
        conn.close()
    
    def store_benchmark_run(self, run_data: Dict, benchmark_results: Dict) -> int:
        """Store a complete benchmark run in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert run metadata
        cursor.execute("""
            INSERT INTO performance_runs 
            (timestamp, commit_sha, branch, rust_version, features, environment_info)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            run_data.get('timestamp', datetime.now().isoformat()),
            run_data.get('commit_sha', ''),
            run_data.get('branch', ''),
            run_data.get('rust_version', ''),
            run_data.get('features', ''),
            json.dumps(run_data.get('environment', {}))
        ))
        
        run_id = cursor.lastrowid
        
        # Insert benchmark results
        for test_name, metrics in benchmark_results.items():
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict):
                    value = metric_data.get('value', metric_data.get('mean', 0))
                    std_dev = metric_data.get('std_dev', metric_data.get('stddev', None))
                    unit = metric_data.get('unit', '')
                    sample_count = metric_data.get('sample_count', metric_data.get('samples', None))
                else:
                    value = metric_data
                    std_dev = None
                    unit = ''
                    sample_count = None
                
                cursor.execute("""
                    INSERT INTO benchmark_results
                    (run_id, test_name, metric_name, value, unit, std_dev, sample_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (run_id, test_name, metric_name, value, unit, std_dev, sample_count))
        
        conn.commit()
        conn.close()
        
        return run_id
    
    def get_historical_data(self, days: int = 30) -> pd.DataFrame:
        """Retrieve historical performance data"""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        query = """
            SELECT 
                pr.timestamp,
                pr.commit_sha,
                pr.branch,
                pr.rust_version,
                pr.features,
                br.test_name,
                br.metric_name,
                br.value,
                br.unit,
                br.std_dev,
                br.sample_count
            FROM performance_runs pr
            JOIN benchmark_results br ON pr.id = br.run_id
            WHERE pr.timestamp >= ?
            ORDER BY pr.timestamp, br.test_name, br.metric_name
        """
        
        df = pd.read_sql_query(query, conn, params=[cutoff_date])
        conn.close()
        
        return df
    
    def store_regression_alert(self, run_id: int, test_name: str, severity: str, 
                             change_percent: float, p_value: float = None):
        """Store a regression alert"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO regression_alerts
            (run_id, test_name, severity, change_percent, p_value, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (run_id, test_name, severity, change_percent, p_value, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()

class CriterionParser:
    """Parser for Criterion.rs benchmark output"""
    
    @staticmethod
    def parse_criterion_output(criterion_dir: str) -> Dict:
        """Parse Criterion.rs benchmark results"""
        criterion_path = Path(criterion_dir)
        
        if not criterion_path.exists():
            logger.error(f"Criterion directory not found: {criterion_dir}")
            return {}
        
        results = {}
        
        # Look for benchmark directories
        for benchmark_dir in criterion_path.iterdir():
            if not benchmark_dir.is_dir():
                continue
            
            # Look for base/estimates.json files
            estimates_file = benchmark_dir / "base" / "estimates.json"
            if estimates_file.exists():
                try:
                    with open(estimates_file, 'r') as f:
                        estimates = json.load(f)
                    
                    test_name = benchmark_dir.name
                    results[test_name] = {
                        'mean_time_ns': estimates.get('mean', {}).get('point_estimate', 0),
                        'std_dev_ns': estimates.get('mean', {}).get('standard_error', 0),
                        'median_time_ns': estimates.get('median', {}).get('point_estimate', 0),
                        'throughput_ops_per_sec': 1e9 / estimates.get('mean', {}).get('point_estimate', 1) if estimates.get('mean', {}).get('point_estimate', 0) > 0 else 0
                    }
                    
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse {estimates_file}: {e}")
        
        return results
    
    @staticmethod
    def run_benchmarks(features: str = "all-features") -> str:
        """Run Criterion benchmarks and return the output directory"""
        logger.info(f"Running benchmarks with features: {features}")
        
        cmd = ["cargo", "bench"]
        if features != "default":
            cmd.extend([f"--{features}"])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Benchmarks completed successfully")
            
            # Return the criterion output directory
            return "./target/criterion"
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Benchmark execution failed: {e}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            raise

class PerformanceAnalyzer:
    """Main performance analysis class"""
    
    def __init__(self, db_path: str = "performance_history.db"):
        self.db = PerformanceDatabase(db_path)
        self.parser = CriterionParser()
    
    def analyze_current_run(self, criterion_dir: str, run_metadata: Dict = None) -> Dict:
        """Analyze a current benchmark run"""
        logger.info("Parsing benchmark results...")
        benchmark_results = self.parser.parse_criterion_output(criterion_dir)
        
        if not benchmark_results:
            logger.warning("No benchmark results found")
            return {}
        
        # Prepare run metadata
        if run_metadata is None:
            run_metadata = {
                'timestamp': datetime.now().isoformat(),
                'commit_sha': self._get_git_commit(),
                'branch': self._get_git_branch(),
                'rust_version': self._get_rust_version(),
                'features': 'all-features',
                'environment': self._get_environment_info()
            }
        
        # Store in database
        run_id = self.db.store_benchmark_run(run_metadata, benchmark_results)
        logger.info(f"Stored benchmark run with ID: {run_id}")
        
        # Perform regression analysis
        regression_analysis = self.detect_regressions(benchmark_results)
        
        # Store any regression alerts
        for test_name, analysis in regression_analysis.items():
            if analysis.get('is_regression'):
                self.db.store_regression_alert(
                    run_id, test_name, analysis['severity'],
                    analysis['change_percent'], analysis.get('p_value')
                )
        
        return {
            'run_id': run_id,
            'benchmark_results': benchmark_results,
            'regression_analysis': regression_analysis,
            'run_metadata': run_metadata
        }
    
    def detect_regressions(self, current_results: Dict, baseline_days: int = 7) -> Dict:
        """Detect performance regressions compared to recent baseline"""
        # Get recent historical data for baseline
        historical_df = self.db.get_historical_data(days=baseline_days)
        
        if historical_df.empty:
            logger.warning("No historical data available for regression detection")
            return {}
        
        regression_analysis = {}
        
        for test_name, metrics in current_results.items():
            # Get baseline statistics
            test_data = historical_df[historical_df['test_name'] == test_name]
            
            if test_data.empty:
                continue
            
            # Focus on mean execution time
            baseline_times = test_data[test_data['metric_name'] == 'mean_time_ns']['value']
            
            if baseline_times.empty:
                continue
            
            baseline_mean = baseline_times.mean()
            baseline_std = baseline_times.std()
            current_mean = metrics.get('mean_time_ns', 0)
            
            if baseline_mean == 0:
                continue
            
            # Calculate change
            change_percent = ((current_mean - baseline_mean) / baseline_mean) * 100
            
            # Simple statistical test (could be enhanced with proper t-test)
            z_score = (current_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0
            
            # Determine if this is a significant regression
            is_regression = change_percent > 10.0 and abs(z_score) > 2.0  # 10% threshold with 2-sigma significance
            
            severity = 'severe' if change_percent > 25.0 else 'moderate' if change_percent > 10.0 else 'minor'
            
            regression_analysis[test_name] = {
                'baseline_mean': baseline_mean,
                'current_mean': current_mean,
                'change_percent': change_percent,
                'z_score': z_score,
                'is_regression': is_regression,
                'severity': severity,
                'baseline_samples': len(baseline_times)
            }
        
        return regression_analysis
    
    def generate_static_plots(self, output_dir: str = "./performance_plots"):
        """Generate static performance plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get historical data
        df = self.db.get_historical_data(days=30)
        
        if df.empty:
            logger.warning("No data available for plotting")
            return
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'])
        
        # Plot 1: Performance trends over time
        plt.figure(figsize=(15, 10))
        
        # Get unique test names
        test_names = df['test_name'].unique()
        
        for i, test_name in enumerate(test_names[:6]):  # Limit to 6 tests for readability
            plt.subplot(2, 3, i + 1)
            
            test_data = df[(df['test_name'] == test_name) & (df['metric_name'] == 'mean_time_ns')]
            
            if not test_data.empty:
                plt.plot(test_data['datetime'], test_data['value'], 'o-', alpha=0.7)
                plt.title(f"{test_name} - Mean Time")
                plt.ylabel("Time (ns)")
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_trends.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Performance distribution
        plt.figure(figsize=(12, 8))
        
        recent_data = df[df['metric_name'] == 'mean_time_ns'].copy()
        recent_data = recent_data.groupby('test_name')['value'].last().reset_index()
        
        if not recent_data.empty:
            plt.bar(range(len(recent_data)), recent_data['value'])
            plt.xticks(range(len(recent_data)), recent_data['test_name'], rotation=45, ha='right')
            plt.ylabel("Mean Time (ns)")
            plt.title("Current Performance by Test")
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Static plots saved to {output_dir}")
    
    def export_html_report(self, output_file: str):
        """Export a comprehensive HTML report"""
        df = self.db.get_historical_data(days=30)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .metric {{ background-color: #e9ecef; padding: 10px; margin: 10px 0; border-radius: 3px; }}
                .regression {{ background-color: #f8d7da; color: #721c24; }}
                .improvement {{ background-color: #d4edda; color: #155724; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Performance Analysis Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Analysis Period: Last 30 days</p>
                <p>Total Data Points: {len(df)}</p>
            </div>
        """
        
        if not df.empty:
            # Summary statistics
            html_content += "<h2>Summary Statistics</h2>"
            
            summary_stats = df.groupby(['test_name', 'metric_name'])['value'].agg(['mean', 'std', 'min', 'max'])
            
            html_content += "<table>"
            html_content += "<tr><th>Test</th><th>Metric</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Max</th></tr>"
            
            for (test_name, metric_name), row in summary_stats.iterrows():
                html_content += f"<tr><td>{test_name}</td><td>{metric_name}</td>"
                html_content += f"<td>{row['mean']:.2f}</td><td>{row['std']:.2f}</td>"
                html_content += f"<td>{row['min']:.2f}</td><td>{row['max']:.2f}</td></tr>"
            
            html_content += "</table>"
        
        html_content += """
            <h2>Recommendations</h2>
            <ul>
                <li>Continue regular performance monitoring</li>
                <li>Investigate any significant regressions immediately</li>
                <li>Document performance improvements for release notes</li>
                <li>Maintain comprehensive benchmark coverage</li>
            </ul>
            
            <footer>
                <p><em>Generated by scirs2-special Performance Dashboard</em></p>
            </footer>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report exported to {output_file}")
    
    def _get_git_commit(self) -> str:
        """Get current git commit SHA"""
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except:
            return "unknown"
    
    def _get_git_branch(self) -> str:
        """Get current git branch"""
        try:
            result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                                  capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except:
            return "unknown"
    
    def _get_rust_version(self) -> str:
        """Get Rust version"""
        try:
            result = subprocess.run(['rustc', '--version'], 
                                  capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except:
            return "unknown"
    
    def _get_environment_info(self) -> Dict:
        """Get environment information"""
        import platform
        return {
            'os': platform.system(),
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
            'cpu_count': os.cpu_count()
        }

class InteractiveDashboard:
    """Interactive dashboard using Dash/Plotly"""
    
    def __init__(self, analyzer: PerformanceAnalyzer, port: int = 8050):
        if not INTERACTIVE_AVAILABLE:
            raise ImportError("Interactive dashboard requires plotly and dash")
        
        self.analyzer = analyzer
        self.port = port
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = html.Div([
            html.H1("SciRS2 Special Functions Performance Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': 30}),
            
            html.Div([
                html.Div([
                    html.H3("Controls"),
                    html.Label("Days of History:"),
                    dcc.Slider(
                        id='days-slider',
                        min=7,
                        max=90,
                        step=7,
                        value=30,
                        marks={i: str(i) for i in range(7, 91, 14)}
                    ),
                    html.Br(),
                    html.Button('Refresh Data', id='refresh-button', n_clicks=0),
                ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': 20}),
                
                html.Div([
                    dcc.Graph(id='performance-trends'),
                ], style={'width': '80%', 'display': 'inline-block'}),
            ]),
            
            html.Div([
                html.Div([
                    dcc.Graph(id='performance-distribution'),
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='regression-heatmap'),
                ], style={'width': '50%', 'display': 'inline-block'}),
            ]),
            
            html.Div([
                html.H3("Recent Performance Summary"),
                html.Div(id='performance-summary'),
            ], style={'margin': 20}),
            
            dcc.Interval(
                id='interval-component',
                interval=60*1000,  # Update every minute
                n_intervals=0
            )
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        @self.app.callback(
            [Output('performance-trends', 'figure'),
             Output('performance-distribution', 'figure'),
             Output('regression-heatmap', 'figure'),
             Output('performance-summary', 'children')],
            [Input('refresh-button', 'n_clicks'),
             Input('days-slider', 'value'),
             Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n_clicks, days, n_intervals):
            # Get data
            df = self.analyzer.db.get_historical_data(days=days)
            
            if df.empty:
                empty_fig = go.Figure().add_annotation(
                    text="No data available", 
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=20)
                )
                return empty_fig, empty_fig, empty_fig, "No performance data available"
            
            # Convert timestamp
            df['datetime'] = pd.to_datetime(df['timestamp'])
            
            # Performance trends
            trends_fig = go.Figure()
            
            for test_name in df['test_name'].unique():
                test_data = df[(df['test_name'] == test_name) & (df['metric_name'] == 'mean_time_ns')]
                if not test_data.empty:
                    trends_fig.add_trace(go.Scatter(
                        x=test_data['datetime'],
                        y=test_data['value'],
                        mode='lines+markers',
                        name=test_name,
                        line=dict(width=2)
                    ))
            
            trends_fig.update_layout(
                title="Performance Trends Over Time",
                xaxis_title="Date",
                yaxis_title="Time (ns)",
                hovermode='x unified'
            )
            
            # Performance distribution
            recent_data = df[df['metric_name'] == 'mean_time_ns'].copy()
            recent_data = recent_data.sort_values('datetime').groupby('test_name').last().reset_index()
            
            dist_fig = px.bar(
                recent_data,
                x='test_name',
                y='value',
                title="Current Performance Distribution",
                labels={'value': 'Time (ns)', 'test_name': 'Test Name'}
            )
            dist_fig.update_xaxes(tickangle=45)
            
            # Regression heatmap (simplified)
            heatmap_data = df.pivot_table(
                values='value',
                index='test_name',
                columns=df['datetime'].dt.date,
                aggfunc='mean'
            )
            
            if not heatmap_data.empty:
                # Calculate percentage change from first day
                pct_change = heatmap_data.div(heatmap_data.iloc[:, 0], axis=0) - 1
                pct_change = pct_change * 100  # Convert to percentage
                
                heatmap_fig = px.imshow(
                    pct_change.values,
                    x=[str(col) for col in pct_change.columns],
                    y=pct_change.index,
                    color_continuous_scale='RdYlBu_r',
                    title="Performance Change Heatmap (%)"
                )
                heatmap_fig.update_xaxes(tickangle=45)
            else:
                heatmap_fig = go.Figure()
            
            # Performance summary
            summary_stats = df.groupby('test_name')['value'].agg(['mean', 'std', 'count'])
            summary_text = []
            
            for test_name, stats in summary_stats.iterrows():
                summary_text.append(
                    html.Div([
                        html.H4(test_name),
                        html.P(f"Mean: {stats['mean']:.2f} ns"),
                        html.P(f"Std Dev: {stats['std']:.2f} ns"),
                        html.P(f"Samples: {stats['count']}")
                    ], style={'display': 'inline-block', 'margin': 20, 'padding': 10, 'border': '1px solid #ccc'})
                )
            
            return trends_fig, dist_fig, heatmap_fig, summary_text
    
    def run(self):
        """Run the interactive dashboard"""
        logger.info(f"Starting interactive dashboard on port {self.port}")
        self.app.run_server(debug=False, host='0.0.0.0', port=self.port)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Performance Dashboard for scirs2-special")
    
    parser.add_argument('--benchmark-dir', default='./target/criterion',
                       help='Directory containing Criterion benchmark results')
    parser.add_argument('--run-benchmarks', action='store_true',
                       help='Run benchmarks before analysis')
    parser.add_argument('--features', default='all-features',
                       choices=['default', 'all-features', 'simd', 'parallel'],
                       help='Features to use when running benchmarks')
    parser.add_argument('--db-path', default='performance_history.db',
                       help='Path to performance database')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive dashboard')
    parser.add_argument('--port', type=int, default=8050,
                       help='Port for interactive dashboard')
    parser.add_argument('--export-html', 
                       help='Export HTML report to specified file')
    parser.add_argument('--export-plots', default='./performance_plots',
                       help='Directory to export static plots')
    parser.add_argument('--days', type=int, default=30,
                       help='Days of history to analyze')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer(args.db_path)
    
    # Run benchmarks if requested
    if args.run_benchmarks:
        try:
            criterion_dir = analyzer.parser.run_benchmarks(args.features)
            args.benchmark_dir = criterion_dir
        except Exception as e:
            logger.error(f"Failed to run benchmarks: {e}")
            sys.exit(1)
    
    # Analyze current run if benchmark directory exists
    if os.path.exists(args.benchmark_dir):
        try:
            analysis_result = analyzer.analyze_current_run(args.benchmark_dir)
            
            # Print regression summary
            regression_analysis = analysis_result.get('regression_analysis', {})
            regressions = [test for test, analysis in regression_analysis.items() 
                          if analysis.get('is_regression')]
            
            if regressions:
                logger.warning(f"Performance regressions detected in {len(regressions)} tests:")
                for test in regressions:
                    analysis = regression_analysis[test]
                    logger.warning(f"  {test}: {analysis['change_percent']:+.1f}% ({analysis['severity']})")
            else:
                logger.info("No performance regressions detected")
                
        except Exception as e:
            logger.error(f"Failed to analyze benchmark results: {e}")
    
    # Generate static plots
    if args.export_plots:
        try:
            analyzer.generate_static_plots(args.export_plots)
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")
    
    # Export HTML report
    if args.export_html:
        try:
            analyzer.export_html_report(args.export_html)
        except Exception as e:
            logger.error(f"Failed to export HTML report: {e}")
    
    # Start interactive dashboard
    if args.interactive:
        if not INTERACTIVE_AVAILABLE:
            logger.error("Interactive dashboard requires plotly and dash packages")
            sys.exit(1)
        
        try:
            dashboard = InteractiveDashboard(analyzer, args.port)
            dashboard.run()
        except Exception as e:
            logger.error(f"Failed to start interactive dashboard: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()