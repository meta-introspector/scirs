#!/bin/bash

# Fix rand API issues in scirs2-neural
# Convert thread_rng() to rng() and update imports

echo "Fixing rand API issues..."

# Fix thread_rng() function calls to rng()
find . -name "*.rs" -exec sed -i 's/thread_rng()/rng()/g' {} \;

# Fix imports: remove thread_rng from use statements
find . -name "*.rs" -exec sed -i 's/use rand::{thread_rng, Rng};/use rand::Rng;/g' {} \;
find . -name "*.rs" -exec sed -i 's/use rand::{Rng, thread_rng};/use rand::Rng;/g' {} \;
find . -name "*.rs" -exec sed -i 's/use rand::thread_rng;/use rand::rng;/g' {} \;

# Fix multi-line imports
find . -name "*.rs" -exec sed -i '/use rand::{/ {
    s/thread_rng,//g
    s/, thread_rng//g
    s/thread_rng//g
}' {} \;

# Fix specific patterns in imports
find . -name "*.rs" -exec sed -i 's/use rand::{rngs::SmallRng, thread_rng, SeedableRng};/use rand::{rngs::SmallRng, SeedableRng};/g' {} \;

echo "Rand API fixes completed."