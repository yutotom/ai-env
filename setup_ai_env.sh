#!/usr/bin/env bash
set -euo pipefail

# インストーラーを別の作業ディレクトリから実行しても、正しい対象を参照する。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$SCRIPT_DIR/src/ai_env.py"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
COMMAND_PATH="$BIN_DIR/ai-env"

if [[ ! -f "$TARGET" ]]; then
  echo "Error: installation target not found: $TARGET" >&2
  exit 1
fi

mkdir -p "$BIN_DIR"
chmod +x "$TARGET"
ln -sfn "$TARGET" "$COMMAND_PATH"

echo "Installed: $COMMAND_PATH -> $TARGET"

# PATH 未設定時だけ、シェル設定へ追加する内容を案内する。
case ":$PATH:" in
  *":$BIN_DIR:"*)
    ;;
  *)
    echo "Add this to your shell config if ai-env is not found:"
    echo "  export PATH=\"$BIN_DIR:\$PATH\""
    ;;
esac
