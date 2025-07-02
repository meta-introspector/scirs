#!/bin/bash

# Clean duplicate imports

echo "Cleaning duplicate imports..."

for file in $(find . -name "*.rs"); do
    # Remove duplicate use rand::rng; lines
    awk '!seen[$0]++' "$file" > "${file}.tmp" && mv "${file}.tmp" "$file"
done

echo "Duplicate import cleanup completed."