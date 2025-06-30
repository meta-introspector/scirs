#!/usr/bin/env python3
"""
Comprehensive CI Report Generator for scirs2-optim

This script analyzes CI artifacts and generates comprehensive reports for
performance regression, security analysis, cross-platform compatibility,
and integration testing results.
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import re


class CIReportGenerator:
    """Generate comprehensive CI reports from artifact data."""
    
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = Path(artifacts_dir)
        self.report_data = {}
        
    def analyze_artifacts(self) -> Dict[str, Any]:
        """Analyze all CI artifacts and extract relevant data."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "test_results": self._analyze_test_results(),
            "performance": self._analyze_performance_results(),
            "memory_analysis": self._analyze_memory_results(),
            "security": self._analyze_security_results(),
            "compatibility": self._analyze_compatibility_results(),
            "integration": self._analyze_integration_results(),
            "summary": {}
        }
        
        # Generate summary statistics
        report["summary"] = self._generate_summary(report)
        
        return report
    
    def _analyze_test_results(self) -> Dict[str, Any]:
        """Analyze test results from nextest output."""
        test_results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "platforms": {},
            "failures": []
        }
        
        # Look for test result artifacts
        for artifact_dir in self.artifacts_dir.glob("test-results-*"):
            platform = artifact_dir.name.replace("test-results-", "")
            
            # Parse nextest results if available
            nextest_file = artifact_dir / "default" / "junit.xml"
            if nextest_file.exists():
                platform_results = self._parse_junit_xml(nextest_file)
                test_results["platforms"][platform] = platform_results
                
                test_results["total_tests"] += platform_results["total"]
                test_results["passed"] += platform_results["passed"]
                test_results["failed"] += platform_results["failed"]
                test_results["skipped"] += platform_results["skipped"]
                
                if platform_results["failures"]:
                    test_results["failures"].extend(platform_results["failures"])
        
        return test_results
    
    def _analyze_performance_results(self) -> Dict[str, Any]:
        """Analyze performance benchmark results."""
        performance = {
            "benchmarks": {},
            "regressions": [],
            "improvements": [],
            "baseline_comparison": {}
        }
        
        # Look for performance artifacts
        perf_dir = self.artifacts_dir / "performance-results"
        if perf_dir.exists():
            # Parse criterion results
            criterion_dir = perf_dir / "criterion"
            if criterion_dir.exists():
                performance["benchmarks"] = self._parse_criterion_results(criterion_dir)
        
        # Look for regression analysis
        regression_dir = self.artifacts_dir / "regression-analysis-*"
        for reg_dir in self.artifacts_dir.glob("regression-analysis-*"):
            regression_file = reg_dir / "regression_report.json"
            if regression_file.exists():
                with open(regression_file) as f:
                    regression_data = json.load(f)
                    performance["regressions"].extend(regression_data.get("regressions", []))
                    performance["improvements"].extend(regression_data.get("improvements", []))
        
        return performance
    
    def _analyze_memory_results(self) -> Dict[str, Any]:
        """Analyze memory usage and leak detection results."""
        memory = {
            "peak_usage": {},
            "leaks_detected": [],
            "allocation_patterns": {},
            "efficiency_score": 0.0
        }
        
        # Look for memory analysis artifacts
        memory_dir = self.artifacts_dir / "memory-analysis"
        if memory_dir.exists():
            # Parse valgrind reports
            valgrind_reports = list(memory_dir.glob("valgrind_reports/*"))
            if valgrind_reports:
                memory["valgrind_results"] = self._parse_valgrind_reports(valgrind_reports)
            
            # Parse memory profiler results
            memory_profile = memory_dir / "memory_profile.json"
            if memory_profile.exists():
                with open(memory_profile) as f:
                    profile_data = json.load(f)
                    memory.update(profile_data)
        
        return memory
    
    def _analyze_security_results(self) -> Dict[str, Any]:
        """Analyze security audit results."""
        security = {
            "vulnerabilities": [],
            "audit_passed": True,
            "dependency_issues": [],
            "security_score": 100.0
        }
        
        # Look for security artifacts
        security_dir = self.artifacts_dir / "security-analysis"
        if security_dir.exists():
            audit_report = security_dir / "security_audit_report.md"
            if audit_report.exists():
                security_content = audit_report.read_text()
                security["vulnerabilities"] = self._parse_security_report(security_content)
                security["audit_passed"] = len(security["vulnerabilities"]) == 0
        
        return security
    
    def _analyze_compatibility_results(self) -> Dict[str, Any]:
        """Analyze cross-platform compatibility results."""
        compatibility = {
            "platforms": {},
            "overall_score": 0.0,
            "issues": []
        }
        
        # Look for compatibility artifacts
        for compat_dir in self.artifacts_dir.glob("compatibility-*"):
            platform = compat_dir.name.replace("compatibility-", "")
            
            results_file = compat_dir / "compatibility_results" / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    platform_data = json.load(f)
                    compatibility["platforms"][platform] = platform_data
        
        # Calculate overall compatibility score
        if compatibility["platforms"]:
            scores = [p.get("score", 0) for p in compatibility["platforms"].values()]
            compatibility["overall_score"] = sum(scores) / len(scores)
        
        return compatibility
    
    def _analyze_integration_results(self) -> Dict[str, Any]:
        """Analyze ML framework integration test results."""
        integration = {
            "frameworks": {},
            "overall_success": True,
            "failed_integrations": []
        }
        
        # Look for integration artifacts
        integration_dir = self.artifacts_dir / "integration-results"
        if integration_dir.exists():
            results_file = integration_dir / "integration_results.json"
            if results_file.exists():
                with open(results_file) as f:
                    integration_data = json.load(f)
                    integration.update(integration_data)
        
        return integration
    
    def _parse_junit_xml(self, junit_file: Path) -> Dict[str, Any]:
        """Parse JUnit XML test results."""
        # Basic XML parsing for test results
        # In a real implementation, you'd use xml.etree.ElementTree
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "failures": []
        }
        
        try:
            content = junit_file.read_text()
            
            # Extract test counts (simplified regex parsing)
            total_match = re.search(r'tests="(\d+)"', content)
            if total_match:
                results["total"] = int(total_match.group(1))
            
            failures_match = re.search(r'failures="(\d+)"', content)
            if failures_match:
                results["failed"] = int(failures_match.group(1))
            
            skipped_match = re.search(r'skipped="(\d+)"', content)
            if skipped_match:
                results["skipped"] = int(skipped_match.group(1))
            
            results["passed"] = results["total"] - results["failed"] - results["skipped"]
            
            # Extract failure details
            failure_pattern = r'<testcase[^>]*name="([^"]*)"[^>]*>.*?<failure[^>]*>(.*?)</failure>'
            for match in re.finditer(failure_pattern, content, re.DOTALL):
                results["failures"].append({
                    "test_name": match.group(1),
                    "failure_message": match.group(2).strip()
                })
        
        except Exception as e:
            print(f"Error parsing JUnit XML {junit_file}: {e}")
        
        return results
    
    def _parse_criterion_results(self, criterion_dir: Path) -> Dict[str, Any]:
        """Parse Criterion benchmark results."""
        benchmarks = {}
        
        for bench_dir in criterion_dir.iterdir():
            if bench_dir.is_dir():
                bench_name = bench_dir.name
                
                # Look for estimates.json
                estimates_file = bench_dir / "base" / "estimates.json"
                if estimates_file.exists():
                    try:
                        with open(estimates_file) as f:
                            estimates = json.load(f)
                            benchmarks[bench_name] = {
                                "mean": estimates.get("mean", {}),
                                "std_dev": estimates.get("std_dev", {}),
                                "median": estimates.get("median", {})
                            }
                    except Exception as e:
                        print(f"Error parsing benchmark {bench_name}: {e}")
        
        return benchmarks
    
    def _parse_valgrind_reports(self, valgrind_files: List[Path]) -> Dict[str, Any]:
        """Parse Valgrind memory analysis reports."""
        valgrind_results = {
            "total_errors": 0,
            "definitely_lost": 0,
            "possibly_lost": 0,
            "reports": []
        }
        
        for report_file in valgrind_files:
            try:
                content = report_file.read_text()
                
                # Parse memory errors
                error_match = re.search(r'ERROR SUMMARY: (\d+) errors', content)
                if error_match:
                    valgrind_results["total_errors"] += int(error_match.group(1))
                
                # Parse memory leaks
                def_lost_match = re.search(r'definitely lost: ([\d,]+) bytes', content)
                if def_lost_match:
                    bytes_lost = int(def_lost_match.group(1).replace(',', ''))
                    valgrind_results["definitely_lost"] += bytes_lost
                
                poss_lost_match = re.search(r'possibly lost: ([\d,]+) bytes', content)
                if poss_lost_match:
                    bytes_lost = int(poss_lost_match.group(1).replace(',', ''))
                    valgrind_results["possibly_lost"] += bytes_lost
                
                valgrind_results["reports"].append({
                    "file": report_file.name,
                    "summary": self._extract_valgrind_summary(content)
                })
                
            except Exception as e:
                print(f"Error parsing Valgrind report {report_file}: {e}")
        
        return valgrind_results
    
    def _extract_valgrind_summary(self, content: str) -> str:
        """Extract summary from Valgrind report."""
        lines = content.split('\n')
        summary_lines = []
        
        for line in lines:
            if 'HEAP SUMMARY:' in line or 'LEAK SUMMARY:' in line:
                summary_lines.append(line.strip())
            elif line.strip().startswith('definitely lost:') or \
                 line.strip().startswith('indirectly lost:') or \
                 line.strip().startswith('possibly lost:'):
                summary_lines.append(line.strip())
        
        return '\n'.join(summary_lines)
    
    def _parse_security_report(self, content: str) -> List[Dict[str, Any]]:
        """Parse security audit report for vulnerabilities."""
        vulnerabilities = []
        
        # Look for vulnerability patterns in the report
        vuln_pattern = r'(HIGH|MEDIUM|LOW|CRITICAL):\s*(.*?)(?=\n\n|\n[A-Z]+:|\Z)'
        
        for match in re.finditer(vuln_pattern, content, re.DOTALL):
            severity = match.group(1)
            description = match.group(2).strip()
            
            vulnerabilities.append({
                "severity": severity,
                "description": description
            })
        
        return vulnerabilities
    
    def _generate_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall summary of CI results."""
        test_results = report["test_results"]
        performance = report["performance"]
        memory = report["memory_analysis"]
        security = report["security"]
        compatibility = report["compatibility"]
        integration = report["integration"]
        
        summary = {
            "overall_status": "PASS",
            "test_pass_rate": 0.0,
            "performance_regressions": len(performance["regressions"]),
            "security_issues": len(security["vulnerabilities"]),
            "compatibility_score": compatibility["overall_score"],
            "memory_leaks": len(memory["leaks_detected"]),
            "failed_integrations": len(integration["failed_integrations"])
        }
        
        # Calculate test pass rate
        if test_results["total_tests"] > 0:
            summary["test_pass_rate"] = test_results["passed"] / test_results["total_tests"]
        
        # Determine overall status
        if (test_results["failed"] > 0 or 
            summary["performance_regressions"] > 0 or
            summary["security_issues"] > 0 or
            summary["memory_leaks"] > 0 or
            summary["failed_integrations"] > 0):
            summary["overall_status"] = "FAIL"
        elif summary["test_pass_rate"] < 0.95:
            summary["overall_status"] = "UNSTABLE"
        
        return summary
    
    def generate_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """Generate a comprehensive markdown report."""
        summary = report_data["summary"]
        
        md = f"""# CI/CD Report - scirs2-optim

**Generated:** {report_data["timestamp"]}  
**Overall Status:** {summary["overall_status"]}

## 📊 Summary

| Metric | Value | Status |
|--------|-------|--------|
| Test Pass Rate | {summary["test_pass_rate"]:.1%} | {'✅' if summary["test_pass_rate"] > 0.95 else '⚠️'} |
| Performance Regressions | {summary["performance_regressions"]} | {'✅' if summary["performance_regressions"] == 0 else '❌'} |
| Security Issues | {summary["security_issues"]} | {'✅' if summary["security_issues"] == 0 else '❌'} |
| Memory Leaks | {summary["memory_leaks"]} | {'✅' if summary["memory_leaks"] == 0 else '❌'} |
| Compatibility Score | {summary["compatibility_score"]:.1f}% | {'✅' if summary["compatibility_score"] > 90 else '⚠️'} |
| Failed Integrations | {summary["failed_integrations"]} | {'✅' if summary["failed_integrations"] == 0 else '❌'} |

## 🧪 Test Results

"""
        
        test_results = report_data["test_results"]
        md += f"""
**Total Tests:** {test_results["total_tests"]}  
**Passed:** {test_results["passed"]} ✅  
**Failed:** {test_results["failed"]} {'❌' if test_results["failed"] > 0 else ''}  
**Skipped:** {test_results["skipped"]}

### Platform Results
"""
        
        for platform, results in test_results["platforms"].items():
            status = "✅" if results["failed"] == 0 else "❌"
            md += f"- **{platform}**: {results['passed']}/{results['total']} passed {status}\n"
        
        if test_results["failures"]:
            md += "\n### Failed Tests\n"
            for failure in test_results["failures"][:5]:  # Show first 5 failures
                md += f"- `{failure['test_name']}`: {failure['failure_message'][:100]}...\n"
        
        # Performance section
        performance = report_data["performance"]
        md += f"""

## 🚀 Performance Analysis

**Regressions Detected:** {len(performance["regressions"])}  
**Performance Improvements:** {len(performance["improvements"])}

"""
        
        if performance["regressions"]:
            md += "### ⚠️ Performance Regressions\n"
            for regression in performance["regressions"][:3]:
                md += f"- {regression.get('benchmark', 'Unknown')}: {regression.get('degradation', 'N/A')}% slower\n"
        
        if performance["improvements"]:
            md += "### 🎉 Performance Improvements\n"
            for improvement in performance["improvements"][:3]:
                md += f"- {improvement.get('benchmark', 'Unknown')}: {improvement.get('improvement', 'N/A')}% faster\n"
        
        # Memory section
        memory = report_data["memory_analysis"]
        md += f"""

## 🧠 Memory Analysis

**Memory Leaks Detected:** {len(memory["leaks_detected"])}  
**Efficiency Score:** {memory["efficiency_score"]:.1f}%

"""
        
        if "valgrind_results" in memory:
            valgrind = memory["valgrind_results"]
            md += f"""
### Valgrind Results
- **Total Errors:** {valgrind["total_errors"]}
- **Definitely Lost:** {valgrind["definitely_lost"]} bytes
- **Possibly Lost:** {valgrind["possibly_lost"]} bytes
"""
        
        # Security section
        security = report_data["security"]
        md += f"""

## 🔒 Security Analysis

**Audit Status:** {'✅ PASSED' if security["audit_passed"] else '❌ FAILED'}  
**Vulnerabilities Found:** {len(security["vulnerabilities"])}  
**Security Score:** {security["security_score"]:.1f}%

"""
        
        if security["vulnerabilities"]:
            md += "### Security Issues\n"
            for vuln in security["vulnerabilities"][:3]:
                md += f"- **{vuln['severity']}**: {vuln['description'][:100]}...\n"
        
        # Compatibility section
        compatibility = report_data["compatibility"]
        md += f"""

## 🌐 Cross-Platform Compatibility

**Overall Score:** {compatibility["overall_score"]:.1f}%

### Platform Results
"""
        
        for platform, results in compatibility["platforms"].items():
            score = results.get("score", 0)
            status = "✅" if score > 90 else "⚠️" if score > 70 else "❌"
            md += f"- **{platform}**: {score:.1f}% {status}\n"
        
        # Integration section
        integration = report_data["integration"]
        md += f"""

## 🔗 Framework Integration

**Overall Success:** {'✅' if integration["overall_success"] else '❌'}  
**Failed Integrations:** {len(integration["failed_integrations"])}

### Framework Results
"""
        
        for framework, results in integration["frameworks"].items():
            status = "✅" if results.get("success", False) else "❌"
            md += f"- **{framework}**: {status}\n"
        
        md += """

---

*This report was automatically generated by the scirs2-optim CI/CD system.*
"""
        
        return md


def main():
    parser = argparse.ArgumentParser(description="Generate CI report from artifacts")
    parser.add_argument("artifacts_dir", help="Directory containing CI artifacts")
    parser.add_argument("--format", choices=["markdown", "json", "html"], 
                       default="markdown", help="Output format")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.artifacts_dir):
        print(f"Error: Artifacts directory {args.artifacts_dir} does not exist")
        sys.exit(1)
    
    generator = CIReportGenerator(args.artifacts_dir)
    report_data = generator.analyze_artifacts()
    
    if args.format == "json":
        output = json.dumps(report_data, indent=2)
    elif args.format == "markdown":
        output = generator.generate_markdown_report(report_data)
    else:
        print(f"Format {args.format} not yet implemented")
        sys.exit(1)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Report written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()