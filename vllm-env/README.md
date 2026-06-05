# vLLM environment

vLLM は PyTorch / CUDA / Transformers との制約が強いため、root の検証環境とは分けて管理します。

## インストール

CUDA 12.8 の PyTorch index を使う設定です。

```bash
cd vllm-env
uv sync
```

確認:

```bash
uv run python -c "import torch, transformers, vllm; print(torch.__version__); print(transformers.__version__); print(vllm.__version__)"
```

この環境も `pyproject.toml` で `torch-cu128` index を指定しています。
