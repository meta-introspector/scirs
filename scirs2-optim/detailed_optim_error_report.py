#!/usr/bin/env python3
"""
Detailed report of files that use OptimError but don't import it properly
"""
import os
import re
from pathlib import Path

def analyze_file_for_optim_error(file_path):
    """Analyze a file for OptimError usage and import status"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Find OptimError usage patterns
        optim_error_patterns = [
            (r'OptimError::', 'OptimError::'),
            (r'-> OptimError', '-> OptimError'),
            (r'Result<.*OptimError', 'Result<*, OptimError>'),
            (r'return OptimError', 'return OptimError'),
            (r'Err\(OptimError', 'Err(OptimError'),
            (r'Ok\(OptimError', 'Ok(OptimError'),
        ]
        
        usage_lines = []
        for i, line in enumerate(lines, 1):
            for pattern, description in optim_error_patterns:
                if re.search(pattern, line):
                    usage_lines.append((i, line.strip(), description))
        
        if not usage_lines:
            return None
        
        # Check for proper imports
        import_patterns = [
            r'use crate::error::\{.*OptimError.*\}',
            r'use crate::error::OptimError',
            r'use crate::error::\{OptimError',
        ]
        
        has_proper_import = any(re.search(pattern, content) for pattern in import_patterns)
        
        # Check what imports exist
        existing_imports = []
        for i, line in enumerate(lines, 1):
            if re.search(r'use crate::error::', line):
                existing_imports.append((i, line.strip()))
        
        return {
            'file': file_path,
            'usage_lines': usage_lines,
            'has_proper_import': has_proper_import,
            'existing_imports': existing_imports,
            'needs_import': not has_proper_import
        }
        
    except Exception as e:
        return {
            'file': file_path,
            'error': str(e)
        }

def main():
    """Main function to analyze all Rust files"""
    src_dir = Path("/media/kitasan/Backup/scirs/scirs2-linalg/../scirs2-optim/src")
    
    if not src_dir.exists():
        print(f"Directory {src_dir} does not exist")
        return
    
    files_needing_import = []
    all_files_analyzed = 0
    
    # Find all .rs files
    for rs_file in src_dir.rglob("*.rs"):
        all_files_analyzed += 1
        analysis = analyze_file_for_optim_error(rs_file)
        if analysis and analysis.get('needs_import', False):
            files_needing_import.append(analysis)
    
    print(f"# OptimError Import Analysis Report")
    print(f"")
    print(f"- **Total files analyzed**: {all_files_analyzed}")
    print(f"- **Files needing OptimError import**: {len(files_needing_import)}")
    print(f"")
    
    if files_needing_import:
        print("## Files that need OptimError imports:")
        print("")
        
        for analysis in files_needing_import:
            rel_path = analysis['file'].relative_to(src_dir)
            print(f"### `{rel_path}`")
            print("")
            
            if analysis.get('existing_imports'):
                print("**Existing imports:**")
                for line_num, import_line in analysis['existing_imports']:
                    print(f"- Line {line_num}: `{import_line}`")
                print("")
            
            if analysis.get('usage_lines'):
                print("**OptimError usage:**")
                for line_num, line_content, pattern_type in analysis['usage_lines']:
                    print(f"- Line {line_num}: `{line_content}` (pattern: {pattern_type})")
                print("")
            
            print("**Suggested fix:**")
            if analysis.get('existing_imports'):
                # Has some imports, suggest modifying existing
                existing_import = analysis['existing_imports'][0][1]
                if 'Result' in existing_import and '{' in existing_import:
                    suggested = existing_import.replace('Result}', 'OptimError, Result}')
                else:
                    suggested = existing_import.replace('Result', '{OptimError, Result}')
                print(f"- Replace `{existing_import}` with `{suggested}`")
            else:
                # No imports, suggest adding new
                print(f"- Add `use crate::error::{{OptimError, Result}};` at the top of the file")
            print("")
            print("---")
            print("")
    else:
        print("## Result: All files have proper imports!")

if __name__ == "__main__":
    main()