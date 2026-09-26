#!/usr/bin/env bash
set -euo pipefail

# シンボリックリンク経由でも、同梱の Python スクリプトを参照する。
SCRIPT_DIR="$(dirname -- "$(realpath -- "${BASH_SOURCE[0]}")")"

# 実行先プロジェクトの設定を読み込み、更新対象のディレクトリを維持する。
if [[ -f "$PWD/.envrc" ]]; then
  source "$PWD/.envrc"
fi
if [[ -n "${UV_PROJECT_ENVIRONMENT:-${VIRTUAL_ENV:-}}" ]]; then
  VENV_DIR="$(realpath -m -- "${UV_PROJECT_ENVIRONMENT:-$VIRTUAL_ENV}")"
  export VIRTUAL_ENV="$VENV_DIR"
  export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "エラー: uv が見つかりません。uv をインストールし、PATH を設定してください。" >&2
  exit 1
fi

exec uv run --script "$SCRIPT_DIR/ai_env.py" "$@"
