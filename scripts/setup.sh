#!/usr/bin/env bash
# One-time local setup: Databricks CLI, python deps, and a JRE for local Spark.
set -euo pipefail

cd "$(dirname "$0")/.."

ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
step() { printf '\n\033[1m%s\033[0m\n' "$1"; }

step "Databricks CLI"
if command -v databricks >/dev/null 2>&1; then
  ok "already installed — $(databricks --version)"
else
  if command -v brew >/dev/null 2>&1; then
    brew tap databricks/tap
    brew install databricks
    ok "installed $(databricks --version)"
  else
    warn "Homebrew not found. Install the CLI manually:"
    warn "  curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh"
    exit 1
  fi
fi

# The bundle features used here (dashboards as resources, serverless defaults)
# need a reasonably recent CLI. v0.240+ is a safe floor.
step "CLI version check"
ver=$(databricks --version | sed -E 's/^Databricks CLI v?//' | cut -d- -f1)
ok "v${ver}"

step "Python dependencies"
if ! command -v uv >/dev/null 2>&1; then
  warn "uv not found — install with: brew install uv"
  exit 1
fi
uv sync --group dev
ok "synced"

step "Java 17 (for local Spark tests)"
# Homebrew installs openjdk keg-only, so it is normal for it to be absent from
# PATH. tests/conftest.py finds it there and sets JAVA_HOME itself — no symlink
# and no shell profile edit needed.
if command -v java >/dev/null 2>&1; then
  ok "$(java -version 2>&1 | head -1)"
elif [[ -x /opt/homebrew/opt/openjdk@17/bin/java ]]; then
  ok "keg-only at /opt/homebrew/opt/openjdk@17 (tests pick this up automatically)"
else
  warn "No JDK 17. Local Spark tests will skip until: brew install openjdk@17"
  warn "Everything else in this repo works without it."
fi

# Spark's Hadoop layer calls getpwuid() at startup. If macOS DirectoryServices
# is wedged the lookup returns nothing and Spark dies with a confusing
# "failure to login" Kerberos error that has nothing to do with Kerberos.
if ! id -un >/dev/null 2>&1 || [[ "$(id -un)" == "$(id -u)" ]]; then
  warn "macOS cannot resolve your uid to a username (\`id -un\` returns a number)."
  warn "Local Spark, brew and sudo will all fail until this is fixed. Reboot."
fi

step "Next"
cat <<'EOF'
  1. make auth                 log in to your Free Edition workspace
  2. fill in warehouse_id      see .env.example, then set it in databricks.yml
                               or pass --var="warehouse_id=..." on deploy
  3. make ship                 test -> deploy -> run
EOF
