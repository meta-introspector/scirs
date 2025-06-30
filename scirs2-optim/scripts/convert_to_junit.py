#!/usr/bin/env python3
"""
JUnit XML Converter for Performance Regression Results

This script converts performance regression analysis results to JUnit XML format
for integration with CI/CD systems like GitHub Actions, Jenkins, etc.
"""

import argparse
import json
import os
import sys
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from xml.dom import minidom


class JUnitConverter:
    """Convert performance regression results to JUnit XML format."""
    
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
    
    def convert_to_junit(
        self,
        regression_report_path: str,
        output_junit_path: str,
        features: str
    ) -> None:
        """Convert regression report to JUnit XML format."""
        
        self.logger.info(f"Converting regression report to JUnit XML for features: {features}")
        
        # Load regression report data
        regression_data = self._load_json_file(regression_report_path)
        
        if not regression_data:
            self.logger.warning("No regression data found, creating empty test suite")
            regression_data = self._create_empty_regression_data(features)
        
        # Generate JUnit XML
        junit_xml = self._generate_junit_xml(regression_data, features)
        
        # Write JUnit XML to file
        self._write_junit_xml(junit_xml, output_junit_path)
        
        self.logger.info(f"JUnit XML generated: {output_junit_path}")
    
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
    
    def _create_empty_regression_data(self, features: str) -> Dict[str, Any]:
        """Create empty regression data structure."""
        return {
            'status': 'Unknown',
            'features': features,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'regression_count': 0,
            'critical_regressions': 0,
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'regressions': [],
            'metadata': {
                'commit_hash': 'unknown',
                'branch': 'unknown',
                'features': features
            }
        }
    
    def _generate_junit_xml(self, regression_data: Dict[str, Any], features: str) -> ET.Element:
        """Generate JUnit XML from regression data."""
        
        # Extract key metrics
        total_tests = regression_data.get('total_tests', 0)
        failed_tests = regression_data.get('failed_tests', 0)
        regression_count = regression_data.get('regression_count', 0)
        critical_regressions = regression_data.get('critical_regressions', 0)
        
        # Calculate total time (use placeholder if not available)
        total_time = regression_data.get('total_execution_time', 0.0)
        
        # Create root testsuites element
        testsuites = ET.Element('testsuites')
        testsuites.set('name', f'Performance Regression Tests - {features}')
        testsuites.set('tests', str(total_tests))
        testsuites.set('failures', str(failed_tests))
        testsuites.set('errors', '0')  # We don't track errors separately
        testsuites.set('time', str(total_time))
        testsuites.set('timestamp', datetime.now(timezone.utc).isoformat())
        
        # Create testsuite for performance regressions
        testsuite = ET.SubElement(testsuites, 'testsuite')
        testsuite.set('name', f'Performance Regression Detection - {features}')
        testsuite.set('tests', str(total_tests or 1))  # At least 1 test
        testsuite.set('failures', str(failed_tests))
        testsuite.set('errors', '0')
        testsuite.set('time', str(total_time))
        testsuite.set('timestamp', datetime.now(timezone.utc).isoformat())
        testsuite.set('package', 'scirs2_optim.performance')
        
        # Add properties
        properties = ET.SubElement(testsuite, 'properties')
        
        # Add metadata as properties
        metadata = regression_data.get('metadata', {})
        for key, value in metadata.items():
            prop = ET.SubElement(properties, 'property')
            prop.set('name', key)
            prop.set('value', str(value))
        
        # Add performance metrics as properties
        prop = ET.SubElement(properties, 'property')
        prop.set('name', 'regression_count')
        prop.set('value', str(regression_count))
        
        prop = ET.SubElement(properties, 'property')
        prop.set('name', 'critical_regressions')
        prop.set('value', str(critical_regressions))
        
        prop = ET.SubElement(properties, 'property')
        prop.set('name', 'overall_status')
        prop.set('value', regression_data.get('status', 'Unknown'))
        
        # Create test cases from regression data
        if total_tests == 0:
            # Create a single overall test case if no individual tests
            self._add_overall_test_case(testsuite, regression_data, features)
        else:
            # Add individual regression test cases
            self._add_regression_test_cases(testsuite, regression_data, features)
        
        return testsuites
    
    def _add_overall_test_case(
        self,
        testsuite: ET.Element,
        regression_data: Dict[str, Any],
        features: str
    ) -> None:
        """Add overall performance regression test case."""
        
        testcase = ET.SubElement(testsuite, 'testcase')
        testcase.set('name', f'Overall Performance Regression Test - {features}')
        testcase.set('classname', 'scirs2_optim.performance.OverallRegression')
        testcase.set('time', str(regression_data.get('total_execution_time', 0.0)))
        
        status = regression_data.get('status', 'Unknown')
        regression_count = regression_data.get('regression_count', 0)
        critical_regressions = regression_data.get('critical_regressions', 0)
        
        # Determine if test passed or failed
        if status == 'Failed' or critical_regressions > 0:
            # Add failure element
            failure = ET.SubElement(testcase, 'failure')
            failure.set('message', f'Performance regression detected: {regression_count} total regressions, {critical_regressions} critical')
            failure.set('type', 'PerformanceRegression')
            
            # Add detailed failure text
            failure_details = self._generate_failure_details(regression_data)
            failure.text = failure_details
            
        elif status == 'Warning':
            # Add system-out for warnings
            system_out = ET.SubElement(testcase, 'system-out')
            system_out.text = f'Performance warnings detected: {regression_count} potential regressions'
        
        # Add system-out with summary
        system_out = ET.SubElement(testcase, 'system-out')
        summary = self._generate_test_summary(regression_data)
        system_out.text = summary
    
    def _add_regression_test_cases(
        self,
        testsuite: ET.Element,
        regression_data: Dict[str, Any],
        features: str
    ) -> None:
        """Add individual regression test cases."""
        
        regressions = regression_data.get('regressions', [])
        benchmarks = regression_data.get('benchmarks', {})
        
        # If we have detailed benchmark data, create test cases for each
        if benchmarks:
            for benchmark_name, benchmark_data in benchmarks.items():
                self._add_benchmark_test_case(testsuite, benchmark_name, benchmark_data, features)
        
        # If we have regression list, add those as separate test cases
        elif regressions:
            for i, regression in enumerate(regressions):
                self._add_regression_item_test_case(testsuite, regression, i, features)
        
        # Add overall summary test case
        self._add_summary_test_case(testsuite, regression_data, features)
    
    def _add_benchmark_test_case(
        self,
        testsuite: ET.Element,
        benchmark_name: str,
        benchmark_data: Dict[str, Any],
        features: str
    ) -> None:
        """Add test case for individual benchmark."""
        
        testcase = ET.SubElement(testsuite, 'testcase')
        testcase.set('name', f'Benchmark: {benchmark_name}')
        testcase.set('classname', f'scirs2_optim.performance.benchmarks.{self._sanitize_classname(benchmark_name)}')
        testcase.set('time', str(benchmark_data.get('execution_time', 0.0)))
        
        # Check if this benchmark has a regression
        status = benchmark_data.get('status', 'passed')
        change_percent = benchmark_data.get('change_percentage', 0.0)
        
        if status == 'failed' or status == 'regression':
            # Add failure element
            failure = ET.SubElement(testcase, 'failure')
            failure.set('message', f'Performance regression in {benchmark_name}: {change_percent:+.1f}% change')
            failure.set('type', 'BenchmarkRegression')
            
            # Add detailed failure information
            failure_text = f"""
Benchmark: {benchmark_name}
Status: {status}
Performance Change: {change_percent:+.1f}%
Current Value: {benchmark_data.get('current_value', 'unknown')}
Baseline Value: {benchmark_data.get('baseline_value', 'unknown')}
Threshold Exceeded: {benchmark_data.get('threshold_exceeded', False)}
Severity: {benchmark_data.get('severity', 'unknown')}
"""
            failure.text = failure_text.strip()
        
        # Add system-out with benchmark details
        system_out = ET.SubElement(testcase, 'system-out')
        system_out.text = json.dumps(benchmark_data, indent=2)
    
    def _add_regression_item_test_case(
        self,
        testsuite: ET.Element,
        regression: Dict[str, Any],
        index: int,
        features: str
    ) -> None:
        """Add test case for individual regression item."""
        
        regression_name = regression.get('name', f'Regression_{index}')
        
        testcase = ET.SubElement(testsuite, 'testcase')
        testcase.set('name', f'Regression: {regression_name}')
        testcase.set('classname', f'scirs2_optim.performance.regressions.{self._sanitize_classname(regression_name)}')
        testcase.set('time', str(regression.get('execution_time', 0.0)))
        
        # Add failure element for regression
        failure = ET.SubElement(testcase, 'failure')
        failure.set('message', regression.get('message', f'Performance regression detected in {regression_name}'))
        failure.set('type', regression.get('type', 'PerformanceRegression'))
        
        # Add detailed failure information
        failure_details = f"""
Regression Details:
Name: {regression_name}
Type: {regression.get('type', 'unknown')}
Severity: {regression.get('severity', 'unknown')}
Change: {regression.get('change_percentage', 0.0):+.1f}%
Threshold: {regression.get('threshold', 'unknown')}
Description: {regression.get('description', 'No description available')}
"""
        failure.text = failure_details.strip()
        
        # Add system-out with full regression data
        system_out = ET.SubElement(testcase, 'system-out')
        system_out.text = json.dumps(regression, indent=2)
    
    def _add_summary_test_case(
        self,
        testsuite: ET.Element,
        regression_data: Dict[str, Any],
        features: str
    ) -> None:
        """Add summary test case."""
        
        testcase = ET.SubElement(testsuite, 'testcase')
        testcase.set('name', f'Performance Summary - {features}')
        testcase.set('classname', 'scirs2_optim.performance.Summary')
        testcase.set('time', str(regression_data.get('total_execution_time', 0.0)))
        
        # Add system-out with overall summary
        system_out = ET.SubElement(testcase, 'system-out')
        summary = self._generate_test_summary(regression_data)
        system_out.text = summary
    
    def _sanitize_classname(self, name: str) -> str:
        """Sanitize name for use as Java classname."""
        # Replace invalid characters with underscores
        sanitized = ''.join(c if c.isalnum() else '_' for c in name)
        # Ensure it starts with a letter
        if sanitized and not sanitized[0].isalpha():
            sanitized = 'Test_' + sanitized
        return sanitized or 'UnknownTest'
    
    def _generate_failure_details(self, regression_data: Dict[str, Any]) -> str:
        """Generate detailed failure information."""
        
        details = f"""
Performance Regression Analysis Results:

Overall Status: {regression_data.get('status', 'Unknown')}
Total Regressions: {regression_data.get('regression_count', 0)}
Critical Regressions: {regression_data.get('critical_regressions', 0)}
Features Tested: {regression_data.get('features', 'unknown')}
Timestamp: {regression_data.get('timestamp', 'unknown')}

"""
        
        # Add regression details if available
        regressions = regression_data.get('regressions', [])
        if regressions:
            details += "Detected Regressions:\n"
            for i, regression in enumerate(regressions[:5], 1):  # Limit to first 5
                details += f"{i}. {regression.get('name', 'Unknown')}: {regression.get('message', 'No details')}\n"
            
            if len(regressions) > 5:
                details += f"... and {len(regressions) - 5} more regressions\n"
        
        # Add benchmark failures if available
        benchmarks = regression_data.get('benchmarks', {})
        failed_benchmarks = [name for name, data in benchmarks.items() if data.get('status') == 'failed']
        if failed_benchmarks:
            details += f"\nFailed Benchmarks ({len(failed_benchmarks)}):\n"
            for benchmark in failed_benchmarks[:5]:  # Limit to first 5
                benchmark_data = benchmarks[benchmark]
                change = benchmark_data.get('change_percentage', 0.0)
                details += f"- {benchmark}: {change:+.1f}% change\n"
            
            if len(failed_benchmarks) > 5:
                details += f"... and {len(failed_benchmarks) - 5} more failed benchmarks\n"
        
        return details.strip()
    
    def _generate_test_summary(self, regression_data: Dict[str, Any]) -> str:
        """Generate test summary information."""
        
        summary = f"""
Performance Regression Test Summary:
====================================

Overall Status: {regression_data.get('status', 'Unknown')}
Features: {regression_data.get('features', 'unknown')}
Timestamp: {regression_data.get('timestamp', 'unknown')}

Test Results:
- Total Tests: {regression_data.get('total_tests', 0)}
- Passed: {regression_data.get('passed_tests', 0)}
- Failed: {regression_data.get('failed_tests', 0)}

Regression Analysis:
- Total Regressions: {regression_data.get('regression_count', 0)}
- Critical Regressions: {regression_data.get('critical_regressions', 0)}
- Warning Level Regressions: {regression_data.get('warning_regressions', 0)}

Performance Metrics:
- Execution Time: {regression_data.get('total_execution_time', 0.0):.3f}s
- Analysis Duration: {regression_data.get('analysis_duration', 0.0):.3f}s

Metadata:
- Commit: {regression_data.get('metadata', {}).get('commit_hash', 'unknown')}
- Branch: {regression_data.get('metadata', {}).get('branch', 'unknown')}
- CI Build: {regression_data.get('metadata', {}).get('ci_build_id', 'unknown')}
"""
        
        return summary.strip()
    
    def _write_junit_xml(self, junit_xml: ET.Element, output_path: str) -> None:
        """Write JUnit XML to file with proper formatting."""
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to string with pretty formatting
        rough_string = ET.tostring(junit_xml, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        # Remove empty lines
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        formatted_xml = '\n'.join(lines)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_xml)


def main():
    """Main entry point for the JUnit converter."""
    parser = argparse.ArgumentParser(description='Convert performance regression results to JUnit XML')
    parser.add_argument('--regression-report', required=True,
                       help='Path to regression analysis report JSON file')
    parser.add_argument('--output-junit', required=True,
                       help='Output path for JUnit XML file')
    parser.add_argument('--features', required=True,
                       help='Feature set being tested')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Configuration
    config = {
        'include_system_out': True,
        'include_properties': True,
        'max_failure_details': 1000,  # Max chars in failure details
    }
    
    # Create converter and convert to JUnit XML
    converter = JUnitConverter(config)
    
    try:
        converter.convert_to_junit(
            args.regression_report,
            args.output_junit,
            args.features
        )
        
        print(" JUnit XML conversion completed successfully")
        print(f"=Ä JUnit XML file: {args.output_junit}")
        
    except Exception as e:
        logging.error(f"JUnit conversion failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()