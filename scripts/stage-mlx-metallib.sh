#!/usr/bin/env bash
set -euo pipefail

# Stage the MLX Metal shader library (mlx.metallib) so Tauri bundles it into
# Echo.app/Contents/MacOS — the directory MLX's runtime "colocated library"
# search (get_colocated_mtllib_path) looks in.
#
# Why this is needed:
#   mlx-rs / mlx-sys compile the Metal kernels into mlx.metallib at build time
#   and bake the *build directory's* absolute path into the binary as the
#   default METAL_PATH. That path exists on the build machine, so dev builds and
#   the CI runner work — but on an installed/distributed app it does not, so MLX
#   aborts at engine init with:
#       MLX error: Failed to load the default metallib. library not found ...
#   which takes the whole app down on startup (the default model is MLX-backed).
#   Placing mlx.metallib next to the executable inside the bundle makes MLX find
#   it everywhere.
#
# This script runs twice in a Tauri build:
#
#   1. `--placeholder` mode, as part of `beforeBuildCommand` (BEFORE cargo build):
#      creates empty stand-in files so that tauri-build's `externalBin` existence
#      check — which runs inside the cargo build script — passes. Without this the
#      build aborts with `resource path 'binaries/mlx.metallib-...' doesn't exist`
#      on a clean checkout (CI), because the real metallib does not exist yet.
#      If a real metallib is already present (incremental local rebuild) it is
#      staged instead of a placeholder.
#
#   2. default (real) mode, as `beforeBundleCommand` (AFTER cargo build, before
#      bundling): the cargo build has now generated the metallib via mlx-sys, so
#      the real file is copied over the placeholder. The .app bundle has not been
#      assembled/signed yet, so the staged file is picked up as an externalBin and
#      signed + notarized as part of the normal Tauri flow. In this mode a missing
#      metallib is a hard error.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${ROOT_DIR}/src-tauri/target"
BIN_DIR="${ROOT_DIR}/src-tauri/binaries"

PLACEHOLDER_MODE=0
if [[ "${1:-}" == "--placeholder" ]]; then
  PLACEHOLDER_MODE=1
fi

# Tauri exposes the resolved target triple to before* hooks. Default to Apple
# Silicon, the only supported target.
TRIPLE="${TAURI_ENV_TARGET_TRIPLE:-aarch64-apple-darwin}"

# Find the metallib produced by the mlx-sys build, preferring a release tree and
# the newest match. `-nt` keeps this dependency-free (no stat/xargs portability
# concerns); the `*/release/*` test inside [[ ]] is a pattern match, not a
# filesystem glob.
find_metallib() {
  local require_release="$1" # "1" => only accept paths under a release tree
  local best=""
  local f
  while IFS= read -r f; do
    if [[ "${require_release}" == "1" && "${f}" != */release/* ]]; then
      continue
    fi
    if [[ -z "${best}" || "${f}" -nt "${best}" ]]; then
      best="${f}"
    fi
  done < <(find "${TARGET_DIR}" -name 'mlx.metallib' -type f 2>/dev/null)
  printf '%s' "${best}"
}

metallib="$(find_metallib 1)"
if [[ -z "${metallib}" ]]; then
  # Fall back to any build profile (e.g. local debug builds).
  metallib="$(find_metallib 0)"
fi

mkdir -p "${BIN_DIR}"

if [[ -z "${metallib}" || ! -f "${metallib}" ]]; then
  if [[ "${PLACEHOLDER_MODE}" == "1" ]]; then
    # Pre-build: the metallib is generated later by cargo. Create empty stand-ins
    # so tauri-build's externalBin existence check passes; beforeBundleCommand
    # overwrites them with the real metallib before bundling.
    for name in mlx default; do
      dest="${BIN_DIR}/${name}.metallib-${TRIPLE}"
      [[ -f "${dest}" ]] || : > "${dest}"
      echo "stage-mlx-metallib: placeholder -> ${dest}"
    done
    exit 0
  fi
  echo "stage-mlx-metallib: could not find mlx.metallib under ${TARGET_DIR}" >&2
  echo "stage-mlx-metallib: the app must be compiled (mlx-sys built) before bundling." >&2
  exit 1
fi

echo "stage-mlx-metallib: source ${metallib}"

# externalBin resolves a config name "binaries/<name>.metallib" to the on-disk
# file "binaries/<name>.metallib-<triple>" (tauri appends "-<triple>" to the end
# of the path; on non-Windows no extension is added — see tauri_utils
# resources::external_binaries) and bundles it back as "Contents/MacOS/
# <name>.metallib". Stage both names MLX may request (C++ builds use "mlx",
# Swift-style builds use "default").
for name in mlx default; do
  dest="${BIN_DIR}/${name}.metallib-${TRIPLE}"
  cp -f "${metallib}" "${dest}"
  echo "stage-mlx-metallib: staged -> ${dest}"
done
