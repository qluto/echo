#!/bin/bash
# Build script for MLX-ASR-Engine sidecar binary
# Creates a standalone binary that can be bundled with the Tauri application.
#
# Requirements:
#   - Python 3.11+ (ARM native on Apple Silicon)
#   - mlx, mlx-audio installed
#   - PyInstaller

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/../src-tauri/binaries"

echo "=== Building MLX-ASR-Engine Sidecar ==="
echo ""

# Check for required tools
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed"
    exit 1
fi

# Ensure output directory exists
mkdir -p "${OUTPUT_DIR}"

cd "${SCRIPT_DIR}"

# Install PyInstaller if not available
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "Installing PyInstaller..."
    pip3 install pyinstaller
fi

# Verify mlx_audio is installed
echo "Verifying dependencies..."
if ! python3 -c "from mlx_audio.transcribe import transcribe; print('mlx_audio: OK')" 2>/dev/null; then
    echo "Error: mlx_audio is not installed or not working"
    echo "Install with: pip install mlx-audio"
    exit 1
fi

# Determine platform suffix
if [[ "$(uname)" == "Darwin" ]]; then
    if [[ "$(uname -m)" == "arm64" ]]; then
        PLATFORM_SUFFIX="aarch64-apple-darwin"
    else
        PLATFORM_SUFFIX="x86_64-apple-darwin"
    fi
elif [[ "$(uname)" == "Linux" ]]; then
    PLATFORM_SUFFIX="x86_64-unknown-linux-gnu"
else
    PLATFORM_SUFFIX="x86_64-pc-windows-msvc"
fi

OUTPUT_NAME="mlx-asr-engine-${PLATFORM_SUFFIX}"

echo "Platform: ${PLATFORM_SUFFIX}"
echo "Output: ${OUTPUT_DIR}/${OUTPUT_NAME}"
echo ""

# Run PyInstaller
echo "Building with PyInstaller..."
pyinstaller \
    --onefile \
    --name "${OUTPUT_NAME}" \
    --hidden-import mlx \
    --hidden-import mlx.core \
    --hidden-import mlx.nn \
    --hidden-import mlx_audio \
    --hidden-import mlx_audio.transcribe \
    --hidden-import numpy \
    --hidden-import soundfile \
    --hidden-import librosa \
    --collect-all mlx \
    --collect-all mlx_audio \
    --target-arch arm64 \
    --strip \
    --noupx \
    --distpath "${OUTPUT_DIR}" \
    --workpath "${SCRIPT_DIR}/build" \
    --specpath "${SCRIPT_DIR}" \
    engine.py

# Cleanup
rm -rf "${SCRIPT_DIR}/build"
rm -f "${SCRIPT_DIR}/${OUTPUT_NAME}.spec"

# Show result
echo ""
echo "=== Build Complete ==="
echo "Binary: ${OUTPUT_DIR}/${OUTPUT_NAME}"
echo "Size: $(du -h "${OUTPUT_DIR}/${OUTPUT_NAME}" | cut -f1)"
echo ""
echo "To include in Tauri, add to tauri.conf.json:"
echo '  "bundle": {'
echo '    "externalBin": ["binaries/mlx-asr-engine"]'
echo '  }'
