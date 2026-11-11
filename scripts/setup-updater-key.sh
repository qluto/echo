#!/usr/bin/env bash
# Create the auto-update signing key, write its public key into tauri.conf.json,
# and register the private key and its password as GitHub secrets.
# The private key stays at ~/.tauri/echo.key and the password in the login
# keychain ("echo updater signing key"). Back the key up: without it, installed
# copies of Echo can no longer be updated automatically.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEY="${HOME}/.tauri/echo.key"
SERVICE="echo updater signing key"

if [ -e "${KEY}" ]; then
  echo "Using the existing key: ${KEY}"
  PW=$(security find-generic-password -s "${SERVICE}" -a "${USER}" -w)
else
  mkdir -p "${HOME}/.tauri"
  PW=$(openssl rand -base64 24)
  (cd "${ROOT}" && npx tauri signer generate --ci -p "${PW}" -w "${KEY}" >/dev/null)
  chmod 600 "${KEY}"
  security add-generic-password -U -s "${SERVICE}" -a "${USER}" -w "${PW}"
  echo "Created ${KEY} (password in the keychain item \"${SERVICE}\")"
fi

node -e '
  const fs = require("fs");
  const [conf, pub] = process.argv.slice(1);
  const c = JSON.parse(fs.readFileSync(conf, "utf8"));
  c.plugins.updater.pubkey = fs.readFileSync(pub, "utf8").trim();
  fs.writeFileSync(conf, JSON.stringify(c, null, 2) + "\n");
' "${ROOT}/src-tauri/tauri.conf.json" "${KEY}.pub"
echo "Wrote the public key into src-tauri/tauri.conf.json"

gh secret set TAURI_SIGNING_PRIVATE_KEY < "${KEY}"
printf '%s' "${PW}" | gh secret set TAURI_SIGNING_PRIVATE_KEY_PASSWORD
echo "Registered the GitHub secrets"
