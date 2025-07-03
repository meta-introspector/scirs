#!/usr/bin/env python3

import subprocess
import re

# Run cargo check and capture output
result = subprocess.run(['cargo', 'check'], 
                       capture_output=True, 
                       text=True, 
                       cwd='/media/kitasan/Backup/scirs/scirs2-optimize')

output = result.stderr

# Parse the output for unused imports
lines = output.split('\n')
file_imports = {}

i = 0
while i < len(lines):
    line = lines[i]
    if 'unused import:' in line:
        # Look for the file path in the next few lines
        for j in range(i-5, i+5):
            if j >= 0 and j < len(lines) and '-->' in lines[j]:
                file_path = lines[j].split('-->')[1].strip().split(':')[0]
                # Extract the import name
                import_match = re.search(r'unused import: `([^`]+)`', line)
                if import_match:
                    import_name = import_match.group(1)
                    if file_path not in file_imports:
                        file_imports[file_path] = []
                    file_imports[file_path].append(import_name)
                break
    i += 1

# Print results
print("UNUSED IMPORTS IN SCIRS2-OPTIMIZE CRATE:")
print("=" * 50)

for file_path, imports in sorted(file_imports.items()):
    print(f"\nFile: {file_path}")
    for import_name in imports:
        print(f"  - {import_name}")

print(f"\nTotal files with unused imports: {len(file_imports)}")
total_imports = sum(len(imports) for imports in file_imports.values())
print(f"Total unused imports: {total_imports}")