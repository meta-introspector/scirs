#!/bin/bash
# Script to fix all double underscores in scirs2-special module names

echo "Fixing double underscores in scirs2-special..."

# Fix all occurrences in .rs files
find . -name "*.rs" -type f -exec sed -i 's/scirs2__special/scirs2_special/g' {} \;
find . -name "*.rs" -type f -exec sed -i 's/scirs2__spatial/scirs2_spatial/g' {} \;
find . -name "*.rs" -type f -exec sed -i 's/scirs2__core/scirs2_core/g' {} \;
find . -name "*.rs" -type f -exec sed -i 's/num__complex/num_complex/g' {} \;

echo "Done! All double underscores should now be fixed."
echo "Run 'cargo check' to verify everything compiles correctly."