# vLLM environment

vLLM は PyTorch / CUDA / Transformers との制約が強いため、root の検証環境とは分けて管理します。

## インストール

vLLM が標準で要求する PyTorch / CUDA / Transformers の組み合わせをそのまま解決します。root 環境の PyTorch / Transformers とは独立しており、特定の PyTorch CUDA index は指定しません。

```bash
cd vllm-env
uv sync
```

確認:

```bash
uv run python -c "import torch, transformers, vllm; print(torch.__version__); print(transformers.__version__); print(vllm.__version__)"
```

Python は `.python-version` と `requires-python = ">=3.12,<3.13"` により CPython 3.12 に限定しています。
