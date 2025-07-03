#!/usr/bin/env python3
"""
Check for files that use OptimError but don't import it properly
"""
import os
import re
from pathlib import Path

def check_file_for_missing_imports(file_path):
    """Check if a file uses OptimError but doesn't import it"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if file uses OptimError
        optim_error_patterns = [
            r'OptimError::',
            r'-> OptimError',
            r'Result<.*OptimError',
            r'return OptimError',
            r'Err\(OptimError',
            r'Ok\(OptimError',
        ]
        
        uses_optim_error = any(re.search(pattern, content) for pattern in optim_error_patterns)
        
        if not uses_optim_error:
            return False, "No OptimError usage"
        
        # Check if it has proper import
        import_patterns = [
            r'use crate::error::\{.*OptimError.*\}',
            r'use crate::error::OptimError',
            r'use crate::error::\{OptimError',
        ]
        
        has_proper_import = any(re.search(pattern, content) for pattern in import_patterns)
        
        if not has_proper_import:
            return True, "Uses OptimError but missing import"
        
        return False, "Has proper import"
        
    except Exception as e:
        return False, f"Error reading file: {e}"

def main():
    """Main function to check all Rust files"""
    src_dir = Path("/media/kitasan/Backup/scirs/scirs2-linalg/../scirs2-optim/src")
    
    if not src_dir.exists():
        print(f"Directory {src_dir} does not exist")
        return
    
    missing_imports = []
    
    # Find all .rs files
    for rs_file in src_dir.rglob("*.rs"):
        needs_import, reason = check_file_for_missing_imports(rs_file)
        if needs_import:
            missing_imports.append((rs_file, reason))
    
    if missing_imports:
        print(f"Found {len(missing_imports)} files that need OptimError imports:")
        print("=" * 80)
        for file_path, reason in missing_imports:
            rel_path = file_path.relative_to(src_dir)
            print(f"{rel_path}: {reason}")
    else:
        print("All files that use OptimError have proper imports!")

if __name__ == "__main__":
    main()