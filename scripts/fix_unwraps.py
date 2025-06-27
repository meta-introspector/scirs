#!/usr/bin/env python3
"""
Script to help identify and fix unwrap() calls in Rust code.
This script finds patterns of unwrap() usage and suggests replacements.
"""

import os
import re
import argparse
from typing import List, Tuple, Dict
from collections import defaultdict

# Common unwrap patterns and their suggested replacements
UNWRAP_PATTERNS = {
    # Array/slice access
    r'(\w+)\[(\w+)\]': 'Use .get() with proper bounds checking',
    
    # Option unwrap
    r'\.unwrap\(\)': 'Replace with ? operator or .ok_or()',
    
    # Arithmetic operations that might panic
    r'(\w+)\s*/\s*(\w+)': 'Use safe_divide() for division',
    r'\.sqrt\(\)': 'Use safe_sqrt() for square root',
    r'\.ln\(\)': 'Use safe_log() for logarithm',
    r'\.log\(\)': 'Use safe_log() for logarithm',
    r'\.log10\(\)': 'Use safe_log10() for base-10 logarithm',
    r'\.powf?\(': 'Use safe_pow() for power operations',
    
    # Type conversions
    r'as f64': 'Use .try_into() or F::from() for safe conversion',
    r'as f32': 'Use .try_into() or F::from() for safe conversion',
    r'as usize': 'Use .try_into() for safe conversion',
    
    # Common unwrap scenarios
    r'\.parse\(\)\.unwrap\(\)': 'Use .parse().map_err(|e| Error::ParseError(e))?',
    r'\.to_owned\(\)\.unwrap\(\)': 'Check if to_owned() can fail here',
    r'Array\d+::from.*\.unwrap\(\)': 'Use proper error handling for array creation',
}

def find_unwraps_in_file(filepath: str) -> List[Tuple[int, str, str]]:
    """Find all unwrap() calls in a file and suggest fixes."""
    findings = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except:
        return findings
    
    for line_num, line in enumerate(lines, 1):
        # Skip comments and test code
        if line.strip().startswith('//') or '#[test]' in line or '#[cfg(test)]' in line:
            continue
            
        # Find direct unwrap() calls
        if '.unwrap()' in line:
            # Try to identify the context
            suggestion = "Replace with proper error handling"
            
            # Check for specific patterns
            for pattern, fix in UNWRAP_PATTERNS.items():
                if re.search(pattern, line):
                    suggestion = fix
                    break
            
            # Special cases
            if 'thread_rng().gen_range' in line:
                suggestion = "Use rng().random_range() for rand 0.9.0+"
            elif 'Array' in line and 'from' in line:
                suggestion = "Handle array creation errors properly"
            elif '.get(' in line and ').unwrap()' in line:
                suggestion = "Use .get().ok_or(Error::IndexOutOfBounds)?"
            elif 'env::var' in line:
                suggestion = "Use .unwrap_or_default() or proper error for env vars"
                
            findings.append((line_num, line.strip(), suggestion))
        
        # Find panic-prone operations even without unwrap
        for pattern in ['/ ', '.sqrt()', '.ln()', '.log()', '.powf(']:
            if pattern in line and '.unwrap()' not in line:
                if pattern == '/ ':
                    # Check if it's actually division
                    if re.search(r'[^/]\s*/\s*[^/]', line) and '//' not in line:
                        findings.append((
                            line_num, 
                            line.strip(), 
                            "Division without zero check - use safe_divide()"
                        ))
                else:
                    findings.append((
                        line_num,
                        line.strip(),
                        f"Mathematical operation {pattern} without validation"
                    ))
    
    return findings

def analyze_crate(crate_path: str) -> Dict[str, List[Tuple[int, str, str]]]:
    """Analyze all Rust files in a crate."""
    results = defaultdict(list)
    
    for root, _, files in os.walk(crate_path):
        # Skip target and other non-source directories
        if 'target' in root or '.git' in root:
            continue
            
        for file in files:
            if file.endswith('.rs'):
                filepath = os.path.join(root, file)
                findings = find_unwraps_in_file(filepath)
                if findings:
                    relative_path = os.path.relpath(filepath, crate_path)
                    results[relative_path] = findings
    
    return results

def generate_fix_report(results: Dict[str, List[Tuple[int, str, str]]]) -> str:
    """Generate a markdown report of findings."""
    report = ["# Unwrap() Usage Report\n"]
    
    total_unwraps = sum(len(findings) for findings in results.values())
    report.append(f"Total unwrap() calls and unsafe operations found: {total_unwraps}\n")
    
    # Group by suggestion type
    suggestion_counts = defaultdict(int)
    for findings in results.values():
        for _, _, suggestion in findings:
            suggestion_counts[suggestion] += 1
    
    report.append("## Summary by Type\n")
    for suggestion, count in sorted(suggestion_counts.items(), key=lambda x: -x[1]):
        report.append(f"- {suggestion}: {count} occurrences")
    
    report.append("\n## Detailed Findings\n")
    
    for file, findings in sorted(results.items()):
        report.append(f"\n### {file}\n")
        report.append(f"{len(findings)} issues found:\n")
        
        for line_num, code, suggestion in findings:
            report.append(f"- Line {line_num}: `{code[:80]}{'...' if len(code) > 80 else ''}`")
            report.append(f"  - **Fix**: {suggestion}")
    
    return '\n'.join(report)

def generate_safe_wrapper_template(crate_name: str) -> str:
    """Generate a template for safe wrapper functions."""
    return f"""// Template for {crate_name}/src/safe_ops.rs

use crate::error::{{{crate_name.replace('-', '_').title()}Error, {crate_name.replace('-', '_').title()}Result}};
use num_traits::{{Float, Zero}};
use std::fmt::Display;

/// Safe division with zero checking
pub fn safe_divide<T: Float + Display>(num: T, denom: T) -> {crate_name.replace('-', '_').title()}Result<T> {{
    if denom.abs() < T::epsilon() {{
        return Err({crate_name.replace('-', '_').title()}Error::DomainError(
            format!("Division by zero or near-zero: {{}} / {{}}", num, denom)
        ));
    }}
    
    let result = num / denom;
    if !result.is_finite() {{
        return Err({crate_name.replace('-', '_').title()}Error::ComputationError(
            format!("Division produced non-finite result: {{:?}}", result)
        ));
    }}
    
    Ok(result)
}}

/// Safe square root with domain checking
pub fn safe_sqrt<T: Float + Display>(value: T) -> {crate_name.replace('-', '_').title()}Result<T> {{
    if value < T::zero() {{
        return Err({crate_name.replace('-', '_').title()}Error::DomainError(
            format!("Cannot compute sqrt of negative value: {{}}", value)
        ));
    }}
    
    Ok(value.sqrt())
}}

// Add more safe operations as needed...
"""

def main():
    parser = argparse.ArgumentParser(description='Find and report unwrap() usage in Rust code')
    parser.add_argument('path', help='Path to crate or workspace')
    parser.add_argument('--output', '-o', help='Output report file', default='unwrap_report.md')
    parser.add_argument('--template', '-t', help='Generate safe wrapper template', action='store_true')
    
    args = parser.parse_args()
    
    print(f"Analyzing {args.path}...")
    results = analyze_crate(args.path)
    
    if results:
        report = generate_fix_report(results)
        
        with open(args.output, 'w') as f:
            f.write(report)
        
        print(f"Report written to {args.output}")
        print(f"Found {sum(len(f) for f in results.values())} issues in {len(results)} files")
        
        if args.template:
            crate_name = os.path.basename(args.path)
            template = generate_safe_wrapper_template(crate_name)
            template_file = f"{crate_name}_safe_ops_template.rs"
            
            with open(template_file, 'w') as f:
                f.write(template)
            
            print(f"Safe wrapper template written to {template_file}")
    else:
        print("No unwrap() calls or unsafe operations found!")

if __name__ == '__main__':
    main()