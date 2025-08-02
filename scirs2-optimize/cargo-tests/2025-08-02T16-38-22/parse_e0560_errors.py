#!/usr/bin/env python3
"""
Parse E0560 errors from cargo compile output and generate structured analysis
"""
import json
import re
from collections import defaultdict

def parse_e0560_errors(json_file):
    """Extract and categorize all E0560 errors"""
    errors = []
    field_patterns = defaultdict(int)
    file_locations = defaultdict(list)
    
    with open(json_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                message = data.get('message', {})
                code_info = message.get('code', {})
                if (data.get('reason') == 'compiler-message' and 
                    code_info and code_info.get('code') == 'E0560'):
                    
                    message = data['message']
                    spans = message.get('spans', [])
                    
                    for span in spans:
                        if span.get('is_primary', False):
                            file_name = span['file_name']
                            line_start = span['line_start']
                            
                            # Extract field name from error message
                            error_text = message['message']
                            field_match = re.search(r'has no field named `([^`]+)`', error_text)
                            if field_match:
                                field_name = field_match.group(1)
                                
                                # Extract suggested field name
                                suggested = None
                                if 'children' in message:
                                    for child in message['children']:
                                        if child.get('level') == 'help':
                                            for child_span in child.get('spans', []):
                                                if 'suggested_replacement' in child_span:
                                                    suggested = child_span['suggested_replacement']
                                
                                error_info = {
                                    'file': file_name,
                                    'line': line_start,
                                    'wrong_field': field_name,
                                    'suggested_field': suggested,
                                    'struct_name': error_text.split('`')[1] if '`' in error_text else 'Unknown'
                                }
                                
                                errors.append(error_info)
                                field_patterns[field_name] += 1
                                file_locations[file_name].append(line_start)
                                
            except (json.JSONDecodeError, KeyError):
                continue
    
    return errors, field_patterns, file_locations

def main():
    errors, field_patterns, file_locations = parse_e0560_errors('compile_output.json')
    
    print("=== E0560 STRUCT FIELD ERRORS ANALYSIS ===")
    print(f"Total E0560 errors found: {len(errors)}")
    print()
    
    print("=== FIELD NAME PATTERNS (by frequency) ===")
    for field, count in sorted(field_patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"{field}: {count} occurrences")
    print()
    
    print("=== FILES WITH E0560 ERRORS ===")
    for file_path, lines in sorted(file_locations.items()):
        print(f"{file_path}: {len(lines)} errors at lines {lines}")
    print()
    
    print("=== DETAILED ERROR LOCATIONS ===")
    for i, error in enumerate(errors, 1):
        print(f"{i:2d}. {error['file']}:{error['line']}")
        print(f"    Struct: {error['struct_name']}")
        print(f"    Wrong field: {error['wrong_field']} → Should be: {error['suggested_field']}")
        print()
    
    # Print highest impact patterns for targeted fixing
    print("=== HIGHEST IMPACT PATTERNS FOR FIXING ===")
    high_impact = [(field, count) for field, count in field_patterns.items() if count >= 3]
    for field, count in sorted(high_impact, key=lambda x: x[1], reverse=True):
        print(f"Pattern '{field}': {count} errors")
        matching_errors = [e for e in errors if e['wrong_field'] == field]
        for error in matching_errors:
            print(f"  - {error['file']}:{error['line']}")
        print()

if __name__ == '__main__':
    main()