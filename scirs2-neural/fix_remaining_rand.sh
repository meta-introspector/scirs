#!/bin/bash

# Script to find and fix remaining compilation issues

echo "Checking for files with syntax errors..."

# Find all Rust files and check them individually
total_files=0
error_files=0

for file in $(find src -name "*.rs"); do
    total_files=$((total_files + 1))
    
    # Try to parse the file syntax
    if ! rustc --crate-type lib "$file" --allow warnings >/dev/null 2>&1; then
        echo "Syntax error in: $file"
        error_files=$((error_files + 1))
    fi
done

echo "Found $error_files files with syntax errors out of $total_files total files"

# List files that commonly have missing closing braces
echo "Checking for common missing brace patterns..."

grep -l "impl.*{[^}]*$" src/**/*.rs | head -10

echo "Done."