#!/bin/bash

# Script to fix all format! calls with positional arguments to use inline variables

# Find all files with format! calls that need fixing
FILES=$(rg -l "format!\s*\(\s*\"[^\"]*\{\}[^\"]*\"\s*," /media/kitasan/Backup/scirs/scirs2-core/src --type rust)

# Counter for tracking fixes
TOTAL_FIXES=0

echo "Starting to fix format! calls in scirs2-core..."

for FILE in $FILES; do
    echo "Processing: $FILE"
    
    # Count format! calls in this file
    CALLS=$(rg -c "format!\s*\(\s*\"[^\"]*\{\}[^\"]*\"\s*," "$FILE" 2>/dev/null || echo 0)
    
    if [ "$CALLS" -gt 0 ]; then
        echo "  Found $CALLS format! calls to fix"
        TOTAL_FIXES=$((TOTAL_FIXES + CALLS))
    fi
done

echo "Total format! calls found: $TOTAL_FIXES"

# Let's create a comprehensive sed-based fix for common patterns
echo "Applying fixes..."

# Fix simple single-variable format calls
find /media/kitasan/Backup/scirs/scirs2-core/src -name "*.rs" -exec sed -i 's/format!("\([^"]*\){}\([^"]*\)", \([^)]*\))/format!("\1{\3}\2")/g' {} \;

echo "Phase 1 complete - fixed simple single-variable format calls"

# Now let's handle more complex cases manually for the most critical files
echo "Manual fixes for complex cases..."
