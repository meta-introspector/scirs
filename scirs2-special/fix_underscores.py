#!/usr/bin/env python3
"""Fix double underscores in module names."""

import os
import re

def fix_file(filepath):
    """Fix double underscores in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace all double underscores in module names
        content = content.replace('scirs2__special', 'scirs2_special')
        content = content.replace('scirs2__spatial', 'scirs2_spatial')
        content = content.replace('scirs2__core', 'scirs2_core')
        content = content.replace('scirs2__datasets', 'scirs2_datasets')
        content = content.replace('num__complex', 'num_complex')
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def process_directory(directory):
    """Process all .rs files in directory recursively."""
    fixed_files = []
    
    for root, dirs, files in os.walk(directory):
        # Skip target directory
        if 'target' in root:
            continue
            
        for file in files:
            if file.endswith('.rs'):
                filepath = os.path.join(root, file)
                if fix_file(filepath):
                    fixed_files.append(filepath)
                    print(f"Fixed: {filepath}")
    
    return fixed_files

if __name__ == "__main__":
    directory = "/media/kitasan/Backup/scirs/scirs2-special"
    print(f"Fixing double underscores in {directory}")
    print("=" * 50)
    
    fixed_files = process_directory(directory)
    
    print(f"\nTotal files fixed: {len(fixed_files)}")
    if fixed_files:
        print("\n✅ Successfully fixed all double underscores!")
    else:
        print("\n✅ No double underscores found or all already fixed.")