#!/bin/bash

# Add proper rand::rng imports where needed

echo "Adding rand::rng imports..."

# Find files that use rng() but don't have the import
for file in $(find . -name "*.rs" -exec grep -l "rng()" {} \;); do
    # Check if file already has rand::rng import
    if ! grep -q "use rand::rng;" "$file"; then
        # Check if file has rng() usage
        if grep -q "rng()" "$file"; then
            # Add import after the first use statement or at the top
            if grep -q "^use " "$file"; then
                # Add after the last use statement
                sed -i '/^use /a use rand::rng;' "$file"
            else
                # Add at the top of the file after any comments/attributes
                sed -i '1i use rand::rng;' "$file"
            fi
            echo "Added rand::rng import to $file"
        fi
    fi
done

echo "Import fixes completed."