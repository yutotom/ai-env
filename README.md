# uv-env-test

`uv` を使い、Linux 上の NVIDIA GPU 向け AI Python 環境を構築・検証するプロジェクトです。

PyTorch / Transformers / TRL / FlashAttention 2 の互換構成を選びます。FlashAttention は source build を行わず、公開済みの prebuilt wheel を利用します。

| 目的 | 手順 |
| --- | --- |
| リポジトリの固定構成を使う | [固定構成のセットアップ](#固定構成のセットアップ) |
| マシンに合う構成を選ぶ・更新する | [構成の自動選択・更新](#構成の自動選択更新) |
| 別のプロジェクトを構成する | [コマンドのインストール](#別プロジェクトでの利用) |

## 対応環境

| 項目 | 自動設定スクリプトの対応範囲 |
| --- | --- |
| OS / アーキテクチャ | Linux x86_64 / aarch64 |
| Python | 3.12（`>=3.12,<3.13`、wheel は CPython 3.12 用） |
| GPU | NVIDIA Ampere / Ada / Hopper |
| NVIDIA ドライバ | CUDA 12.6 以上に対応すること |
| PyTorch index | `cu126` / `cu128` / `cu130` |
| FlashAttention | 対象環境に合う FlashAttention 2 の公式 wheel が GitHub Releases に存在すること |

Windows、macOS、AMD GPU、CPU のみの環境、CUDA 11 系は対象外です。スクリプトは Turing 以前および Blackwell の GPU では停止します。

### このリポジトリの固定構成

現在の [pyproject.toml](pyproject.toml)、[uv.lock](uv.lock)、[.python-version](.python-version) は Linux x86_64 の NVIDIA GPU 環境向けです。

| 項目 | 設定値 |
| --- | --- |
| Python | 3.12.11 |
| PyTorch | 2.10.0（`cu130`） |
| Transformers | 5.14.1 |
| TRL | 1.9.0 |
| FlashAttention | 2.8.3（CUDA 13 / PyTorch 2.10 / CPython 3.12 / x86_64 / `cxx11abiTRUE`） |

既存の検証環境では NVIDIA ドライバ 580.173.02 を維持する方針です。`torchvision` / `torchaudio` は含めていません。

## セットアップ

### 事前確認

`uv` と NVIDIA ドライバを用意し、`nvidia-smi` が正常に動作することを確認してください。以下のセットアップコマンドはリポジトリのルートから実行します。

```bash
uv --version
nvidia-smi
```

`nvidia-smi` が失敗する場合は、先に NVIDIA ドライバの問題を解消してください。構成の自動選択・更新にはネットワーク接続も必要です。

### 固定構成のセットアップ

[固定構成](#このリポジトリの固定構成)に対応するマシンで実行します。

```bash
uv sync
```

### 構成の自動選択・更新

別の NVIDIA GPU マシンで初めて構成する場合や、依存関係を最新の互換構成へ更新する場合に使用します。

```bash
# 選択結果と依存関係の解決を確認する
./src/ai_env.py --dry-run

# 設定を反映し、uv sync で環境を同期する
./src/ai_env.py
```

スクリプトは GPU・ドライバ情報と公開パッケージを調べ、`pyproject.toml` と `uv.lock` を更新します。選択基準は[自動選択の仕組み](#自動選択の仕組み)を参照してください。

#### dry-run の確認範囲

`--dry-run` は選択した依存構成を一時コピーへ反映し、`uv sync --dry-run` で検証します。解決に失敗すると非ゼロで終了します。

- 対象ディレクトリの設定・lockfile・仮想環境は作成・変更しません。
- GPU・ネットワークへのアクセスと `uv` キャッシュの更新は発生します。
- workspace 設定のあるプロジェクトは未対応で、エラーになります。

### 別プロジェクトでの利用

リポジトリのルートでコマンドをインストールし、対象プロジェクトへ移動して実行します。

```bash
./setup_ai_env.sh
cd /path/to/target-project
ai-env --dry-run
ai-env
```

インストール先は標準で `~/.local/bin/ai-env` です。`ai-env` が見つからない場合は、インストーラーの案内に従って `PATH` を設定してください。コマンドはこのリポジトリの `src/ai_env.py` へのシンボリックリンクです。

`ai-env` はカレントディレクトリを更新対象にします。

- `pyproject.toml` がなければ、ディレクトリ名をプロジェクト名にした最小構成を作成します。
- `.python-version` がなければ、`3.12.11` を指定して作成します。
- 既存ファイルは初期化せず、必要な `[tool.uv.sources]` と `[[tool.uv.index]]` を補います。
- 既存の依存関係・extras・管理対象外の配布元は保持し、`torch`、`transformers`、`trl`、`flash-attn` の通常依存と配布元を更新します。

既存 extras の制約が選択結果と競合する場合は、`uv` の解決エラーに従って制約を調整してください。

### オプション

```bash
# 複数 GPU がある場合、構成判定に使う GPU を指定する（標準は 0）
./src/ai_env.py --gpu-index 1

# PyTorch index を指定する（ドライバの対応範囲内であることが必要）
./src/ai_env.py --torch-cuda-index cu128
```

インストール済みの `ai-env` でも同じオプションを使えます。

## 動作確認

セットアップ後、対象プロジェクトで実行します。

```bash
# Python と PyTorch のバージョン、CUDA の利用可否
uv run python --version
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

# Transformers / TRL のバージョン
uv run python -c "import transformers, trl; print(transformers.__version__); print(trl.__version__)"

# FlashAttention の import 確認
uv run python -c "import flash_attn; print('flash_attn ok')"
```

## 自動選択の仕組み

### バージョンの選択ルール

PyTorch は、安定版 FlashAttention 2 の公式 prebuilt wheel が存在する最大バージョンを選びます。Transformers / TRL は PyPI 上の最新バージョンを選び、`uv` で依存関係を解決します。

### CUDA index と wheel の互換性

PyTorch index は、NVIDIA ドライバが対応する CUDA の上限から選択します。これはインストール済み CUDA Toolkit のバージョンではありません。

| ドライバが対応する CUDA の上限 | PyTorch index |
| --- | --- |
| 12.6 以上、12.8 未満 | `cu126` |
| 12.8 以上、13.0 未満 | `cu128` |
| 13.0 以上 | `cu130` |

FlashAttention wheel は CUDA、PyTorch、Python、platform、C++ ABI が一致するものを選択します。対応する公式 Linux CUDA index では `cxx11abiTRUE` の wheel を使用します。

## 運用上の注意

- FlashAttention は GitHub Releases の wheel URL を直接指定します。PyPI からインストールすると source build に入る場合があります。
- PyTorch を個別に更新すると FlashAttention wheel との ABI 互換性が崩れる可能性があります。更新には自動設定スクリプトを使用してください。

## 開発時の確認

スクリプトは `uv run --script` で起動し、TOML の書式を保持する `tomlkit` と依存指定を解析する `packaging` を独立した環境で利用します。

リポジトリのルートで単体テストを実行します。

```bash
uv run python -m unittest discover -s tests
```

GPU 環境を同期せずにスクリプトの構文を確認するには、次を実行します。

```bash
uv run --no-project python -m py_compile src/ai_env.py
```

## 参考

- [PyTorch のインストール](https://pytorch.org/get-started/locally/)
- [PyTorch releases](https://github.com/pytorch/pytorch/releases)
- [uv と PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [uv の設定リファレンス](https://docs.astral.sh/uv/reference/settings/)
- [Transformers のインストール](https://huggingface.co/docs/transformers/installation)
- [FlashAttention releases](https://github.com/Dao-AILab/flash-attention/releases)
- [PyTorch 2.6 の配布仕様（C++ ABI 移行）](https://pytorch.org/blog/pytorch2-6/)
