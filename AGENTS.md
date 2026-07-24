# AGENTS.md

## このリポジトリの目的

`uv` を使い、NVIDIA GPU 向け AI Python 環境の依存関係と互換性を検証するリポジトリです。

- ルート環境は PyTorch / Transformers / TRL / FlashAttention 2 を扱います。
- `vllm-env/` は vLLM 専用の独立した環境です。
- 対象は Linux 上の NVIDIA GPU 環境です。Windows、macOS、AMD GPU、CPU のみの環境は対象外です。
- 正確な対応バージョンや GPU 世代など、変化し得る情報は `README.md` と現在の設定ファイルを正としてください。

## 言語

- 思考は英語で行い、ユーザーへの説明は日本語で行ってください。
- コード内のコメントと docstring は、周囲の記述に合わせて日本語で書いてください。

## 主なファイル

| パス | 役割 |
| --- | --- |
| `README.md` | 対応環境、セットアップ方法、運用上の注意 |
| `pyproject.toml` | ルート環境の Python・AI ライブラリ・PyTorch index の定義 |
| `uv.lock` | ルート環境の固定済み依存関係 |
| `.python-version` | ルート環境の Python バージョン |
| `scripts/ai_env.py` | GPU と公開パッケージを調査し、互換構成を選択・反映する CLI |
| `tests/test_ai_env.py` | CUDA、platform、FlashAttention wheel の選択ロジックの単体テスト |
| `vllm-env/` | ルートとは分離された vLLM 環境 |

## 作業時の原則

- パッケージ管理とコマンド実行には `uv` を使用してください。
- ルート環境と `vllm-env/` の依存関係を混在させないでください。
- `pyproject.toml` を変更した場合は、対応する `uv.lock` も同じ変更に含めてください。
- `vllm-env/pyproject.toml` を変更した場合は、`vllm-env/uv.lock` のみを更新してください。
- Python はルートでは `>=3.12,<3.13`、`vllm-env` でも CPython 3.12 を前提とします。変更する場合は `.python-version`、`requires-python`、wheel の Python tag を一緒に確認してください。
- FlashAttention は source build を避け、対象環境に合う prebuilt wheel を使用してください。
- FlashAttention wheel の CUDA、PyTorch、Python、platform、C++ ABI の互換性を崩さないでください。
- PyTorch のバージョンと `[tool.uv.sources]` / `[[tool.uv.index]]` の CUDA variant は一体として扱ってください。
- `scripts/ai_env.py` の自動選択結果を手作業の思い込みで上書きせず、GPU、driver、公開 wheel の実態を確認してください。
- `.venv/`、キャッシュ、生成されたコマンド記録はコミット対象にしないでください。
- 既存の未コミット変更を保持し、依頼と無関係なファイルを整形・修正しないでください。

## 実装規約

- Python には型ヒントを付け、公開上重要な関数や判定ロジックには簡潔な docstring またはコメントを書いてください。
- CLI のエラーは、失敗した前提と利用者が取るべき対応が分かる日本語にしてください。
- 外部コマンドは引数のリストで実行し、shell 文字列の組み立てを避けてください。
- ネットワーク、GPU、`nvidia-smi` に依存する処理と、純粋な解析・選択ロジックを分離してください。
- バージョンは文字列で大小比較せず、数値のタプルなどに正規化してください。
- TOML 更新時は対象 section だけを変更し、無関係なユーザー設定を保持してください。
- 対応 GPU、CUDA index、wheel 命名規則を変更した場合は、正常系と非対応系のテストを追加してください。

## 標準コマンド

ルート環境:

```bash
uv sync
uv run python -m unittest discover -s tests
uv run python -m py_compile scripts/ai_env.py
```

自動構成スクリプトの安全な確認:

```bash
./scripts/ai_env.py --dry-run
```

`--dry-run` でも GPU、ネットワーク、`nvidia-smi`、GitHub Releases、PyPI を使用します。これらが利用できない環境では、単体テストを優先し、実機確認が未実施であることを報告してください。

vLLM 環境:

```bash
cd vllm-env
uv sync
uv run python -c "import torch, transformers, vllm; print(torch.__version__); print(transformers.__version__); print(vllm.__version__)"
```

## 変更別の確認

- Python ロジックのみ: 単体テストと `py_compile` を実行してください。
- ルート依存関係: 単体テストに加えて `uv sync` または最低限 `uv lock --check` を実行してください。
- GPU 構成選択: 可能なら `./scripts/ai_env.py --dry-run` を実行し、選択された CUDA index と wheel を確認してください。
- vLLM 依存関係: `vllm-env/` 内で同期し、import とバージョン表示を確認してください。
- README の対応表やコマンド例に影響する変更: コード、設定、README を同時に更新してください。

## 完了時の報告

- 変更したファイルと変更理由を簡潔に示してください。
- 実行した検証コマンドと結果を示してください。
- GPU、ネットワーク、時間、権限などの理由で未実施の確認があれば明記してください。
