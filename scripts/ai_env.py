#!/usr/bin/env python3
"""マシンに合うAI環境を構成する。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import platform
import re
import shlex
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
PYTHON_VERSION_FILE = ROOT / ".python-version"
COMMAND_RECORD = ROOT / "configure_gpu_stack.commands.txt"
FLASH_ATTN_RELEASES_URL = "https://api.github.com/repos/Dao-AILab/flash-attention/releases"
TORCH_INDEXES = ("cu126", "cu128", "cu130")


@dataclass(frozen=True)
class FlashAttnWheel:
    """wheel名から取得した互換性情報。"""

    release: str
    name: str
    url: str
    flash_version: str
    cuda: str
    torch: str
    python_tag: str
    platform_tag: str


def run(
    command: list[str], *, capture: bool = False, record: bool = False
) -> subprocess.CompletedProcess[str]:
    """コマンドを表示し、必要なら再現用の記録にも残して実行する。"""

    rendered = shlex.join(command)
    print("実行コマンド: " + rendered, flush=True)
    if record:
        with COMMAND_RECORD.open("a", encoding="utf-8") as output:
            output.write(rendered + "\n")
    return subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def fail(message: str) -> int:
    print(f"エラー: {message}", file=sys.stderr)
    return 1


def require(condition: bool, message: str) -> None:
    """環境の前提を満たさない場合に、利用者向けの理由を付けて停止する。"""

    if not condition:
        raise RuntimeError(message)


def version_tuple(raw: str) -> tuple[int, ...]:
    """文字列比較で 12.10 < 12.8 となる問題を避ける。"""

    return tuple(int(part) for part in raw.split("."))


def cuda_from_output(output: str) -> str | None:
    match = re.search(r"CUDA Version:\s*([0-9.]+)", output)
    return match.group(1) if match else None


def required_cuda(torch_index: str) -> tuple[int, int]:
    digits = torch_index.removeprefix("cu")
    return int(digits[:-1]), int(digits[-1])


def torch_index_for_cuda(cuda: str) -> str:
    """NVIDIA driverが扱える範囲で最も新しいPyTorch indexを選ぶ。"""

    driver_cuda = version_tuple(cuda)
    for minimum, index in (("13.0", "cu130"), ("12.8", "cu128"), ("12.6", "cu126")):
        if driver_cuda >= version_tuple(minimum):
            return index
    raise RuntimeError(
        f"CUDA 12.6 以上に対応する NVIDIA driver が必要です: driver CUDA={cuda}"
    )


def platform_tag(machine: str | None = None) -> str:
    """実行マシンをFlashAttention wheelのplatform tagへ変換する。"""

    current = (machine or platform.machine()).lower()
    if sys.platform.startswith("linux"):
        if current in {"x86_64", "amd64"}:
            return "linux_x86_64"
        if current in {"aarch64", "arm64"}:
            return "linux_aarch64"
    raise RuntimeError(f"未対応 platform です: {platform.platform()} / {current}")


def parse_flash_attn_wheel(
    release: str, asset: dict[str, object]
) -> FlashAttnWheel | None:
    """GitHub Releasesのasset名から対応CUDA・PyTorch・Pythonを読み取る。"""

    name = str(asset.get("name", ""))
    match = re.fullmatch(
        r"flash_attn-(?P<flash>[^+]+)\+"
        r"(?P<cuda>cu\d+)torch(?P<torch>\d+\.\d+(?:\.\d+)?)"
        r"cxx11abi(?:TRUE|FALSE)-(?P<python>cp\d+)-cp\d+-"
        r"(?P<platform>.+)\.whl",
        name,
    )
    if not match:
        return None
    return FlashAttnWheel(
        release,
        name,
        str(asset["browser_download_url"]),
        match.group("flash"),
        match.group("cuda"),
        match.group("torch"),
        match.group("python"),
        match.group("platform"),
    )


def latest_pypi_version(package: str) -> str:
    with urllib.request.urlopen(
        f"https://pypi.org/pypi/{package}/json", timeout=15
    ) as response:
        return json.load(response)["info"]["version"]


def flash_attention2_releases() -> list[dict[str, object]]:
    request = urllib.request.Request(
        FLASH_ATTN_RELEASES_URL,
        headers={"Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def choose_flash_attn_wheel(torch_index: str) -> FlashAttnWheel:
    """現在のPython・OS・CUDAに一致するFlashAttention 2 wheelを選ぶ。"""

    configured_python = PYTHON_VERSION_FILE.read_text(encoding="utf-8").strip()
    python_match = re.match(r"^(\d+)\.(\d+)", configured_python)
    python_tag = (
        f"cp{python_match.group(1)}{python_match.group(2)}"
        if python_match
        else f"cp{sys.version_info.major}{sys.version_info.minor}"
    )
    # FlashAttention側はcu126/cu128を分けず、CUDA majorだけをwheel名に持つ。
    target_cuda = "cu13" if torch_index == "cu130" else "cu12"
    candidates = [
        wheel
        for release in flash_attention2_releases()
        if str(release.get("tag_name", "")).startswith("v2.")
        for asset in release.get("assets", [])
        if (wheel := parse_flash_attn_wheel(str(release["tag_name"]), asset))
        and wheel.python_tag == python_tag
        and wheel.platform_tag == platform_tag()
        and wheel.cuda == target_cuda
    ]
    if not candidates:
        raise RuntimeError(
            "対応する FlashAttention 2 wheel が見つかりません: "
            f"python={python_tag}, platform={platform_tag()}, cuda={target_cuda}"
        )
    # FlashAttentionの最新版より、対応PyTorchが新しいwheelを優先する。
    return max(
        candidates,
        key=lambda wheel: (
            version_tuple(wheel.torch),
            tuple(int(x) for x in re.findall(r"\d+", wheel.flash_version)),
        ),
    )


def query_gpu(gpu_index: int) -> dict[str, str]:
    """driverのCUDA上限と、指定GPUの能力・メモリをnvidia-smiから取得する。"""

    require(shutil.which("nvidia-smi") is not None, "nvidia-smi が見つかりません")
    # CUDA Versionはinstalled toolkitではなくdriverが対応する上限を表す。
    summary = run(["nvidia-smi"], capture=True, record=True)
    require(
        summary.returncode == 0,
        (summary.stderr or summary.stdout or "nvidia-smi が失敗しました").strip(),
    )
    cuda = cuda_from_output(summary.stdout)
    require(cuda is not None, "NVIDIA driver CUDA version を取得できません")
    details = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ],
        capture=True,
        record=True,
    )
    require(details.returncode == 0, (details.stderr or details.stdout).strip())
    for line in details.stdout.strip().splitlines():
        fields = [part.strip() for part in line.split(",", 4)]
        if int(fields[0]) == gpu_index:
            index, name, driver, memory, capability = fields
            return {
                "index": index,
                "name": name,
                "driver": driver,
                "cuda": cuda,
                "memory": memory,
                "compute_capability": capability,
            }
    raise RuntimeError(f"GPU index {gpu_index} が見つかりません")


def validate_gpu(gpu: dict[str, str], torch_index: str) -> None:
    """GPU世代とdriverが選択したFlashAttention構成を実行できるか確認する。"""

    capability = version_tuple(gpu["compute_capability"])
    require(
        (8, 0) <= capability < (10, 0),
        "FlashAttention 2 はAmpere/Ada/Hopper世代専用です: "
        f"compute capability={gpu['compute_capability']}",
    )
    require(
        version_tuple(gpu["cuda"]) >= required_cuda(torch_index),
        "指定した PyTorch CUDA index を NVIDIA driver がサポートしません: "
        f"driver CUDA={gpu['cuda']}, index={torch_index}",
    )


def replace_section(text: str, header: str, body: str) -> str:
    """TOMLの指定sectionだけを置換し、それ以外のユーザー設定を保持する。"""

    pattern = rf"(?ms)^{re.escape(header)}\n.*?(?=^\[|\Z)"
    require(re.search(pattern, text) is not None, f"{header} section が見つかりません")
    return re.sub(pattern, header + "\n" + body.rstrip() + "\n\n", text, count=1)


def update_pyproject(
    wheel: FlashAttnWheel, torch_index: str, transformers: str, trl: str
) -> None:
    """選択結果を依存関係とPyTorchの専用indexへ反映する。"""

    torch_version = wheel.torch if wheel.torch.count(".") == 2 else wheel.torch + ".0"
    dependencies = (
        "dependencies = [\n"
        f'    "torch=={torch_version}",\n'
        f'    "transformers=={transformers}",\n'
        f'    "trl=={trl}",\n'
        f'    "flash-attn @ {wheel.url}",\n'
        '    "ninja",\n'
        '    "packaging",\n'
        '    "psutil",\n'
        "]"
    )
    text = PYPROJECT.read_text(encoding="utf-8")
    text = re.sub(r"(?ms)^dependencies = \[(?:\n.*?\n)?\]", dependencies, text, count=1)
    # 過去の構成にoptional-dependenciesが残っていても二重定義にしない。
    text = re.sub(
        r"(?ms)\n\[project\.optional-dependencies\]\n.*?(?=^\[|\Z)", "\n", text, count=1
    )
    source = f"torch = {{ index = \"torch-{torch_index}\" }}"
    text = replace_section(text, "[tool.uv.sources]", source)
    index = (
        f'name = "torch-{torch_index}"\n'
        f'url = "https://download.pytorch.org/whl/{torch_index}"\n'
        "explicit = true"
    )
    PYPROJECT.write_text(
        replace_section(text, "[[tool.uv.index]]", index), encoding="utf-8"
    )


def configure(args: argparse.Namespace) -> int:
    """ホストに合う依存バージョンを決定し、uvで環境を同期する。"""

    COMMAND_RECORD.write_text("", encoding="utf-8")
    print(f"コマンド記録: {COMMAND_RECORD}", flush=True)
    try:
        gpu = query_gpu(args.gpu_index)
        torch_index = (
            torch_index_for_cuda(gpu["cuda"])
            if args.torch_cuda_index == "auto"
            else args.torch_cuda_index
        )
        validate_gpu(gpu, torch_index)
        packages = {
            package: latest_pypi_version(package)
            for package in ("torch", "transformers", "trl")
        }
        wheel = choose_flash_attn_wheel(torch_index)
    except (OSError, RuntimeError) as error:
        return fail(str(error))

    print(f".python-version: {PYTHON_VERSION_FILE.read_text().strip()}", flush=True)
    print(f"Python: {sys.version.split()[0]}", flush=True)
    print(f"Platform: {platform.platform()} / {platform.machine()}", flush=True)
    uv = run(["uv", "--version"], capture=True, record=True)
    print(f"uv: {uv.stdout.strip() or '(unknown)'}", flush=True)
    for package, latest in packages.items():
        print(f"PyPI {package}: {latest}", flush=True)
    print(
        f"GPU: {gpu['name']} / compute capability {gpu['compute_capability']} / "
        f"driver {gpu['driver']} / driver CUDA {gpu['cuda']} / {gpu['memory']}",
        flush=True,
    )
    print(f"FlashAttention 2 release: {wheel.release}", flush=True)
    print(f"FlashAttention 2 wheel: {wheel.name}", flush=True)
    print(f"FlashAttention 2 wheel が要求する torch: {wheel.torch}", flush=True)
    print(f"PyTorch index: {torch_index}", flush=True)

    # dry-runではpyproject.tomlを変更せず、uvの解決結果だけを確認する。
    if args.dry_run:
        print("dry-run: pyproject.toml は更新せず、uv sync も実行しません。")
        run(["uv", "sync", "--dry-run"], record=True)
        return 0
    update_pyproject(wheel, torch_index, packages["transformers"], packages["trl"])
    print("pyproject.toml を選択した互換バージョンに更新しました。", flush=True)
    return run(["uv", "sync"], record=True).returncode


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dry-run", action="store_true")
    cli.add_argument(
        "--torch-cuda-index", choices=("auto", *TORCH_INDEXES), default="auto"
    )
    cli.add_argument("--gpu-index", type=int, default=0)
    return cli


def main() -> int:
    try:
        return configure(parser().parse_args())
    except Exception as error:
        print(f"エラー: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
