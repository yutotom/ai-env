# uv-env-test

`uv` を使って Python 環境構築を検証するための最小プロジェクトです。PyTorch / Transformers / TRL / FlashAttention のように Python、CUDA、PyTorch のバージョン制約が絡みやすいパッケージを、ビルドせずに wheel だけで入れる方針で管理します。

## 現在の前提

```bash
cat .python-version
uv run python --version
uv --version
nvidia-smi
```

現在の指定:

- `.python-version`: `3.12`
- `pyproject.toml` の `requires-python`: `>=3.12`
- FlashAttention 2 経路: 通常の `dependencies` に PyTorch CUDA wheel / Transformers / TRL / FlashAttention wheel を直接入れる
- `flash-attn` は GitHub Releases の prebuilt wheel を直接指定し、`no-build-package = ["flash-attn"]` で source build を禁止
- `scripts/configure_gpu_stack.py` は FlashAttention 2 の最新 release にある wheel 名から対応する `torch` version を読み取り、`pyproject.toml` を更新してから同期する

## 自動設定

マシン情報を読み取り、利用する PyTorch index と FlashAttention wheel を選んで `uv sync` します。

```bash
./scripts/configure_gpu_stack.py
```

確認だけ行う場合:

```bash
./scripts/configure_gpu_stack.py --dry-run --allow-no-gpu
```

PyTorch index の CUDA variant を手動で固定する場合:

```bash
./scripts/configure_gpu_stack.py --torch-cuda-index cu128
```

## 手動で同期する場合

FlashAttention 2 をビルドせずに使う構成。`torch-cu128` の部分は、`configure_gpu_stack.py` が選んだ index に合わせます。

```bash
uv sync --no-build-package flash-attn
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
