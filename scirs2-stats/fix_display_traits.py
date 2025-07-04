#!/usr/bin/env python3

import re
import os
import glob

def fix_display_traits_in_file(filepath):
    """Add std::fmt::Display trait bounds to generic types that need them"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Pattern to match where clauses that need Display trait
        # Look for F: Float patterns and add Display if missing
        pattern = r'(\s+F:\s+Float[^{]+?)(\s*,?\s*\n*\s*\{)'
        
        def add_display_if_missing(match):
            where_clause = match.group(1)
            ending = match.group(2)
            
            # Only add if Display not already present
            if 'std::fmt::Display' not in where_clause:
                # Remove trailing comma if present
                where_clause = where_clause.rstrip(',').rstrip()
                # Add Display trait
                where_clause += '\n        + std::fmt::Display'
                
            return where_clause + ending
        
        # Apply the fix
        new_content = re.sub(pattern, add_display_if_missing, content, flags=re.MULTILINE)
        
        # Also handle inline trait bounds like impl<F: Float + ...>
        inline_pattern = r'(impl<F:\s*Float[^>]+?)(\s*>)'
        
        def add_display_inline(match):
            traits = match.group(1)
            ending = match.group(2)
            
            if 'std::fmt::Display' not in traits:
                traits = traits.rstrip() + ' + std::fmt::Display'
                
            return traits + ending
        
        new_content = re.sub(inline_pattern, add_display_inline, new_content)
        
        # Only write if changed
        if new_content != content:
            with open(filepath, 'w') as f:
                f.write(new_content)
            print(f"Updated {filepath}")
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    # Find all Rust files in src directory
    rust_files = glob.glob('/media/kitasan/Backup/scirs/scirs2-stats/src/**/*.rs', recursive=True)
    
    updated_count = 0
    for filepath in rust_files:
        if fix_display_traits_in_file(filepath):
            updated_count += 1
    
    print(f"Updated {updated_count} files")

if __name__ == "__main__":
    main()