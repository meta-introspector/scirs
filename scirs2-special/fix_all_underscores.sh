#!/bin/bash

echo "Fixing all double underscores in scirs2-special module..."

# Find all .rs files and fix double underscores
find . -name "*.rs" -type f | grep -v target | while read file; do
    # Check if file has double underscores
    if grep -q "__" "$file"; then
        echo "Processing: $file"
        
        # Create backup
        cp "$file" "${file}.bak"
        
        # Fix all known patterns
        sed -i 's/scirs2__special/scirs2_special/g' "$file"
        sed -i 's/scirs2__spatial/scirs2_spatial/g' "$file"
        sed -i 's/scirs2__datasets/scirs2_datasets/g' "$file"
        sed -i 's/scirs2__core/scirs2_core/g' "$file"
        sed -i 's/num__complex/num_complex/g' "$file"
        sed -i 's/serde__json/serde_json/g' "$file"
        
        # Check if changes were made
        if ! diff -q "$file" "${file}.bak" > /dev/null; then
            echo "  ✓ Fixed double underscores in $file"
            rm "${file}.bak"
        else
            rm "${file}.bak"
        fi
    fi
done

echo "Done! All double underscores have been fixed."