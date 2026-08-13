#!/usr/bin/env bash
# One-time local setup: Databricks CLI, python deps, and a JRE for local Spark.
set -euo pipefail

cd "$(dirname "$0")/.."

ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
step() { printf '\n\033[1m%s\033[0m\n' "$1"; }

# The CLI is a single Go binary, so install it into ~/.local/bin rather than via
# Homebrew or the official install script. Both of those want sudo or a working
# passwd lookup, and neither is guaranteed — this path needs no privileges and
# leaves nothing to uninstall but one file.
step "Databricks CLI"
if command -v databricks >/dev/null 2>&1; then
  ok "already installed — $(databricks --version)"
else
  bindir="$HOME/.local/bin"
  mkdir -p "$bindir"

  tag=$(curl -fsSL https://api.github.com/repos/databricks/cli/releases/latest \
        | grep -m1 '"tag_name"' | cut -d'"' -f4)
  [[ -n "$tag" ]] || { warn "could not resolve the latest CLI release"; exit 1; }

  case "$(uname -s)/$(uname -m)" in
    Darwin/arm64) plat=darwin_arm64 ;;
    Darwin/*)     plat=darwin_amd64 ;;
    Linux/aarch64|Linux/arm64) plat=linux_arm64 ;;
    Linux/*)      plat=linux_amd64 ;;
    *) warn "unsupported platform $(uname -s)/$(uname -m)"; exit 1 ;;
  esac

  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  curl -fsSL -o "$tmp/cli.zip" \
    "https://github.com/databricks/cli/releases/download/${tag}/databricks_cli_${tag#v}_${plat}.zip"
  unzip -oq "$tmp/cli.zip" -d "$tmp"
  mv -f "$tmp/databricks" "$bindir/databricks"
  chmod +x "$bindir/databricks"

  ok "installed $("$bindir/databricks" --version) to $bindir"
  case ":$PATH:" in
    *":$bindir:"*) : ;;
    *) warn "$bindir is not on PATH — add this to ~/.zshrc:"
       warn "    export PATH=\"\$HOME/.local/bin:\$PATH\"" ;;
  esac
fi

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
#
# Probe by *running* java, not by `command -v`: macOS ships a /usr/bin/java stub
# that exists on PATH and only errors when invoked, so a presence check reports
# success on a machine with no JDK at all.
if java -version >/dev/null 2>&1; then
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
