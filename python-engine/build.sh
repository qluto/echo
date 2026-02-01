#!/bin/bash
# Build script for MLX-ASR-Engine sidecar binary
# Creates a standalone binary that can be bundled with the Tauri application.
#
# Requirements:
#   - Python 3.11 (ARM native on Apple Silicon) - Python 3.13 has PyInstaller issues
#   - mlx, mlx-audio installed
#   - PyInstaller
#
# Usage:
#   Local development: ./build.sh (uses/creates venv automatically)
#   CI: Set CI=true or GITHUB_ACTIONS=true (uses system Python)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/../src-tauri/binaries"
VENV_DIR="${SCRIPT_DIR}/venv"

echo "=== Building MLX-ASR-Engine Sidecar ==="
echo ""

# Detect if running in CI
if [[ -n "${CI}" ]] || [[ -n "${GITHUB_ACTIONS}" ]]; then
    echo "Running in CI environment"
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
else
    echo "Running in local environment"

    # Check for Python 3.11
    if command -v /opt/homebrew/bin/python3.11 &> /dev/null; then
        PYTHON311="/opt/homebrew/bin/python3.11"
    elif command -v python3.11 &> /dev/null; then
        PYTHON311="python3.11"
    else
        echo "Error: Python 3.11 is required but not found"
        echo "Install with: brew install python@3.11"
        exit 1
    fi

    # Create venv if it doesn't exist
    if [[ ! -d "${VENV_DIR}" ]]; then
        echo "Creating virtual environment with Python 3.11..."
        "${PYTHON311}" -m venv "${VENV_DIR}"

        # Install dependencies
        echo "Installing dependencies..."
        "${VENV_DIR}/bin/pip" install --upgrade pip
        "${VENV_DIR}/bin/pip" install -r "${SCRIPT_DIR}/requirements.txt"
        "${VENV_DIR}/bin/pip" install pyinstaller
    fi

    # Use venv Python
    PYTHON_CMD="${VENV_DIR}/bin/python3"
    PIP_CMD="${VENV_DIR}/bin/pip"

    # Verify venv Python version
    VENV_VERSION=$("${PYTHON_CMD}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ "${VENV_VERSION}" != "3.11" ]]; then
        echo "Warning: venv Python is ${VENV_VERSION}, expected 3.11"
        echo "Consider removing venv and re-running: rm -rf ${VENV_DIR}"
    fi
fi

# Check Python version
PYTHON_VERSION=$("${PYTHON_CMD}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: ${PYTHON_VERSION}"
if [[ "${PYTHON_VERSION}" == "3.13" ]]; then
    echo "Warning: Python 3.13 has known issues with PyInstaller. Consider using Python 3.11."
fi

# Ensure output directory exists
mkdir -p "${OUTPUT_DIR}"

cd "${SCRIPT_DIR}"

# Install PyInstaller if not available
if ! "${PYTHON_CMD}" -c "import PyInstaller" 2>/dev/null; then
    echo "Installing PyInstaller..."
    "${PIP_CMD}" install pyinstaller
fi

# Verify mlx_audio is installed
echo "Verifying dependencies..."
if ! "${PYTHON_CMD}" -c "from mlx_audio.stt.utils import load_model; print('mlx_audio: OK')" 2>/dev/null; then
    echo "Error: mlx_audio is not installed or not working"
    echo "Install with: ${PIP_CMD} install mlx-audio"
    exit 1
fi

# Get MLX package path for metallib bundling
# mlx is a namespace package, so we get the path from mlx.core
MLX_PATH=$("${PYTHON_CMD}" -c "import mlx.core; import os; print(os.path.dirname(mlx.core.__file__))")
echo "MLX path: ${MLX_PATH}"

# Verify metallib exists
if [[ ! -f "${MLX_PATH}/lib/mlx.metallib" ]]; then
    echo "Error: mlx.metallib not found at ${MLX_PATH}/lib/mlx.metallib"
    echo "MLX may not be properly installed for Apple Silicon"
    exit 1
fi
echo "Found mlx.metallib: ${MLX_PATH}/lib/mlx.metallib"

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

# Run PyInstaller (use the Python from venv/CI)
echo "Building with PyInstaller..."
"${PYTHON_CMD}" -m PyInstaller \
    --onefile \
    --name "${OUTPUT_NAME}" \
    --hidden-import mlx \
    --hidden-import mlx.core \
    --hidden-import mlx.nn \
    --hidden-import mlx_audio \
    --hidden-import mlx_audio.transcribe \
    --hidden-import mlx_audio.stt \
    --hidden-import mlx_audio.stt.utils \
    --hidden-import numpy \
    --hidden-import soundfile \
    --hidden-import librosa \
    --hidden-import torch \
    --hidden-import silero_vad \
    --hidden-import transformers \
    --collect-all mlx \
    --collect-all mlx_audio \
    --collect-all silero_vad \
    --add-data "${MLX_PATH}/lib/mlx.metallib:mlx/lib" \
    --add-data "${MLX_PATH}/lib/libmlx.dylib:mlx/lib" \
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

# Verify the binary was created
if [[ ! -f "${OUTPUT_DIR}/${OUTPUT_NAME}" ]]; then
    echo "Error: Binary was not created"
    exit 1
fi

# Make executable
chmod +x "${OUTPUT_DIR}/${OUTPUT_NAME}"

# Show result
echo ""
echo "=== Build Complete ==="
echo "Binary: ${OUTPUT_DIR}/${OUTPUT_NAME}"
echo "Size: $(du -h "${OUTPUT_DIR}/${OUTPUT_NAME}" | cut -f1)"
echo ""
echo "Quick test (daemon mode):"
echo "  echo '{\"command\":\"ping\",\"id\":1}' | ${OUTPUT_DIR}/${OUTPUT_NAME} daemon"
echo ""
echo "Quick test (single file):"
echo "  ${OUTPUT_DIR}/${OUTPUT_NAME} single /path/to/audio.wav"
