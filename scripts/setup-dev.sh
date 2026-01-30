#!/bin/bash
# Echo - Development Environment Setup Script
# This script sets up the development environment for the Echo application.

set -e

echo "=== Echo Development Setup ==="
echo ""

# Check for required tools
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is not installed"
        echo "Please install $1 before running this script"
        exit 1
    fi
}

echo "Checking required tools..."
check_command node
check_command npm
check_command cargo
check_command python3

echo "✓ All required tools are installed"
echo ""

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Installing Node.js dependencies..."
npm install

echo ""
echo "Setting up Python environment for ASR engine..."
cd python-engine

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate and install dependencies
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

deactivate
cd "$PROJECT_ROOT"

echo ""
echo "Building Rust dependencies (this may take a few minutes)..."
cd src-tauri
cargo build

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run the application in development mode:"
echo "  npm run tauri:dev"
echo ""
echo "To build for production:"
echo "  npm run tauri:build"
echo ""
echo "Note: You may need to grant accessibility permissions to the app for:"
echo "  - Microphone access (for audio recording)"
echo "  - Accessibility permissions (for keyboard input simulation)"
