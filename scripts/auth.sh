#!/usr/bin/env bash
# Log the Databricks CLI in to a workspace.
#
# Tries OAuth (U2M) first — short-lived tokens, nothing secret on disk. Free
# Edition's OAuth support has been inconsistent, so if that fails this falls back
# to a personal access token, which always works.
set -euo pipefail

cd "$(dirname "$0")/.."

PROFILE="${PROFILE:-DEFAULT}"

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  set -a; source .env; set +a
fi

if [[ -z "${DATABRICKS_HOST:-}" ]]; then
  read -rp "Workspace URL (https://dbc-xxxx.cloud.databricks.com): " DATABRICKS_HOST
fi

printf '\nTrying OAuth login to %s ...\n' "$DATABRICKS_HOST"
if databricks auth login --host "$DATABRICKS_HOST" --profile "$PROFILE"; then
  printf '\n\033[32m✓\033[0m OAuth profile "%s" saved to ~/.databrickscfg\n' "$PROFILE"
else
  cat <<EOF

OAuth did not complete. Falling back to a personal access token.

  1. Open ${DATABRICKS_HOST}/settings/user/developer/access-tokens
  2. Generate a new token
  3. Paste it below (input hidden)

EOF
  read -rsp "Token: " token; echo
  databricks configure --host "$DATABRICKS_HOST" --profile "$PROFILE" <<<"$token"
  printf '\n\033[32m✓\033[0m Token profile "%s" saved to ~/.databrickscfg\n' "$PROFILE"
fi

printf '\nVerifying ...\n'
databricks --profile "$PROFILE" current-user me
