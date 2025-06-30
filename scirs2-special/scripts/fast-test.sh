#!/bin/bash
# Ultra-fast development testing script
# Optimizes compilation speed for rapid iteration

echo "🚀 Ultra-Fast Development Mode"
echo "==============================="

# Enable ultra-fast test mode
export ULTRA_FAST_TESTS=1

# Use fast compilation mode with minimal features
echo "Running tests with ultra-fast configuration..."
RUSTFLAGS="-C opt-level=0 -C debuginfo=0" cargo nextest run --features fast-compile

echo "Done! Tests completed in ultra-fast mode."
echo "For full testing, use: cargo nextest run"