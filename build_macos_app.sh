#!/bin/bash
#
# Build script for Glow Flow Analysis macOS App
#
# Usage: ./build_macos_app.sh
#
# Uses the micromamba glow environment
#

set -e

echo "=========================================="
echo "Glow Flow Analysis - macOS App Builder"
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate micromamba glow environment
echo "Activating micromamba glow environment..."
eval "$(micromamba shell hook --shell bash)"
micromamba activate glow

PYTHON_VERSION=$(python --version 2>&1)
echo "Using $PYTHON_VERSION"

# Install PyInstaller if not present
echo ""
echo "Ensuring PyInstaller is installed..."
pip install pyinstaller --quiet

# Ensure data directories exist
echo ""
echo "Ensuring data directories exist..."
mkdir -p data/uploaded_raw
mkdir -p data/active_learning_data

# Clean previous build
echo ""
echo "Cleaning previous build artifacts..."
rm -rf build dist

# Run PyInstaller
echo ""
echo "Building macOS app with PyInstaller..."
echo "This may take several minutes..."
echo ""

pyinstaller Glow.spec --noconfirm

# Check if build succeeded
if [ -d "dist/Glow.app" ]; then
    echo ""
    echo "=========================================="
    echo "Build successful!"
    echo "=========================================="
    echo ""
    echo "App location: $SCRIPT_DIR/dist/Glow.app"
    echo ""
    echo "To run the app:"
    echo "  open dist/Glow.app"
    echo ""
    echo "To install to Applications folder:"
    echo "  cp -r dist/Glow.app /Applications/"
    echo ""
    
    # Get app size
    APP_SIZE=$(du -sh "dist/Glow.app" | cut -f1)
    echo "App size: $APP_SIZE"
else
    echo ""
    echo "=========================================="
    echo "Build failed!"
    echo "=========================================="
    echo "Check the output above for errors."
    exit 1
fi

echo ""
echo "Done!"
