#!/usr/bin/env bash
set -euo pipefail

load_env_file() {
  local env_file="$1"
  [[ -f "${env_file}" ]] || return 0

  while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip comments/empty lines.
    [[ -z "${line//[[:space:]]/}" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue

    # Support optional "export KEY=VALUE" format.
    line="${line#export }"
    if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      local key="${BASH_REMATCH[1]}"
      local raw="${BASH_REMATCH[2]}"

      # Keep existing process env as highest priority.
      if [[ -n "${!key:-}" ]]; then
        continue
      fi

      # Strip wrapping single/double quotes.
      if [[ "$raw" =~ ^\"(.*)\"$ ]]; then
        raw="${BASH_REMATCH[1]}"
      elif [[ "$raw" =~ ^\'(.*)\'$ ]]; then
        raw="${BASH_REMATCH[1]}"
      fi

      export "${key}=${raw}"
    fi
  done < "${env_file}"
}

if [[ "${OSTYPE:-}" != darwin* ]]; then
  echo "This script is for macOS only." >&2
  exit 1
fi

# Load local env files if present.
load_env_file ".env"
load_env_file ".env.local"

if [[ -z "${APPLE_SIGNING_IDENTITY:-}" ]]; then
  cat >&2 <<'EOF'
APPLE_SIGNING_IDENTITY is not set.

Example:
  export APPLE_SIGNING_IDENTITY="Developer ID Application: Your Name (TEAMID)"
  npm run tauri:build:signed
EOF
  exit 1
fi

if ! security find-identity -v -p codesigning 2>/dev/null | grep -Fq "${APPLE_SIGNING_IDENTITY}"; then
  cat >&2 <<EOF
Configured APPLE_SIGNING_IDENTITY was not found in your keychain:
  ${APPLE_SIGNING_IDENTITY}

Install/import the matching "Developer ID Application" certificate first.
EOF
  exit 1
fi

npm run tauri:build -- "$@"

APP_PATH="$(find src-tauri/target -type d -path "*/release/bundle/macos/Echo.app" | head -n 1 || true)"
if [[ -z "${APP_PATH}" ]]; then
  echo "Could not find built Echo.app under src-tauri/target." >&2
  exit 1
fi

SIGNATURE_LINE="$(codesign -dv --verbose=4 "${APP_PATH}" 2>&1 | grep '^Signature=' || true)"
REQ_LINE="$(codesign -d -r- "${APP_PATH}" 2>&1 | grep '^# designated' || true)"

echo "Built app: ${APP_PATH}"
echo "${SIGNATURE_LINE}"
echo "${REQ_LINE}"

if [[ "${SIGNATURE_LINE}" == *"adhoc"* ]]; then
  cat >&2 <<'EOF'
Build output is ad-hoc signed.
This causes macOS Accessibility permission to be treated as a different app each build.
Ensure your Developer ID certificate is installed and APPLE_SIGNING_IDENTITY matches it.
EOF
  exit 1
fi

if [[ "${REQ_LINE}" == *"cdhash"* ]]; then
  cat >&2 <<'EOF'
Designated requirement is cdhash-based (unstable per build).
Do not distribute/install this build for Accessibility persistence.
EOF
  exit 1
fi

echo "Signed build verification passed."
