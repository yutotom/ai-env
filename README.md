# uv-env-test

`uv` を使って Python 環境構築を検証するための最小プロジェクトです。PyTorch / Transformers / TRL / FlashAttention のように Python、CUDA、PyTorch のバージョン制約が絡みやすいパッケージを、ビルドせずに wheel だけで入れる方針で管理します。

## 現在の前提

```bash
cat .python-version
uv run python --version
uv --version
nvidia-smi
```

対応範囲:

- `.python-version`: `3.12.11`
- `pyproject.toml` の `requires-python`: `>=3.12,<3.13`（FlashAttention wheel が CPython 3.12 固定のため）
- 選択した構成: 現行 NVIDIA driver 580.173.02 を維持
- AI ライブラリ: PyTorch 2.10.0 (`cu130`) / Transformers 5.14.1 / TRL 1.9.0 / FlashAttention 2.8.3
- 固定済み lockfile は NVIDIA GPU / Linux / x86_64 用
- 自動設定スクリプトは Linux x86_64 / aarch64 と Ampere、Ada、Hopper GPU に対応
- NVIDIA driver が対応する CUDA 12.6、12.8、13.0
- PyTorch index は `cu126`、`cu128`、`cu130` から自動選択
- GitHub Releases に対象環境用の FlashAttention 2 wheel が存在すること
- FlashAttention 2 経路: 通常の `dependencies` に PyTorch CUDA wheel / Transformers / TRL / FlashAttention wheel を直接入れる
- `flash-attn` は GitHub Releases の x86_64 prebuilt wheel を直接指定し、source build を行わない
- `scripts/ai_env.py` は FlashAttention 2 wheel に対応する `torch` version と、PyPI 上の最新 Transformers / TRL を選んで同期する
- PyTorch は単体最新版ではなく、安定版 FlashAttention 2 の公式 prebuilt wheel が存在する最大バージョンを選ぶ

Windows、macOS、AMD GPU、CPUのみの環境、CUDA 11 系は対象外です。Turing 以前は FlashAttention 2 の対象外、Blackwell は FlashAttention 4 の対象なので、このスクリプトでは明示的に停止します。

## 最新互換セットへの更新

固定バージョンを更新したい場合のみ使用します。マシン情報と公開パッケージを読み取り、利用する PyTorch index と FlashAttention wheel を選んで `pyproject.toml` と `uv.lock` を更新します。

```bash
./scripts/ai_env.py
```

別の NVIDIA GPU マシンで初めて構成する場合もこのコマンドを使用します。

自動選択:

| Driver が対応する CUDA | PyTorch index | 対象例 |
| --- | --- | --- |
| 12.6 | `cu126` | R560 系 |
| 12.8 | `cu128` | R570 系 |
| 13.0 以上 | `cu130` | R580 系以降 |

複数 GPU がある場合、構成判定に使う GPU を指定できます。

```bash
./scripts/ai_env.py --gpu-index 1
```

GPU環境で確認だけ行う場合:

```bash
./scripts/ai_env.py --dry-run
```

PyTorch index の CUDA variant を手動で固定する場合:

```bash
./scripts/ai_env.py --torch-cuda-index cu128
```

## 確認

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
uv run python -c "import transformers, trl; print(transformers.__version__); print(trl.__version__)"
```

FlashAttention 2:

```bash
uv run python -c "import flash_attn; print('flash_attn ok')"
```

## 注意

- FlashAttention 2 の wheel と PyTorch は ABI が合う必要があります。PyPI 上の最新 `torch` ではなく、選択した FlashAttention 2 wheel 名の `torch2.x` に合わせます。
- 2026-07-24 時点では PyTorch 単体最新版は 2.13.0 ですが、安定版 FlashAttention 2 の公式 wheel がある最大バージョンは 2.10.0 です。
- NVIDIA 580.173.02 は現行の Production Branch / Certified driver として維持します。610.43.03 は New Feature Branch のため、この構成では更新しません。
- この検証環境では `torchvision` / `torchaudio` は入れません。
- `flash-attn` の PyPI 配布は source distribution が中心なので、PyPI からそのまま入れるとビルドに入りやすいです。このプロジェクトでは GitHub Releases の wheel URL を使います。
- `nvidia-smi` が失敗する場合、Python 環境を作れても GPU 実行はできません。先に NVIDIA ドライバを直してください。

## 参考

- PyTorch: https://pytorch.org/get-started/locally/
- PyTorch releases: https://github.com/pytorch/pytorch/releases
- uv と PyTorch: https://docs.astral.sh/uv/guides/integration/pytorch/
- uv conflicts / no-build-package: https://docs.astral.sh/uv/reference/settings/
- Transformers: https://huggingface.co/docs/transformers/installation
- FlashAttention releases: https://github.com/Dao-AILab/flash-attention/releases
