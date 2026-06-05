#!/usr/bin/env python3
"""Select a FlashAttention 2 wheel and run a matching uv sync command."""

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
COMMAND_RECORD = ROOT / "configure_gpu_stack.commands.txt"
FLASH_ATTN_RELEASES_URL = "https://api.github.com/repos/Dao-AILab/flash-attention/releases"


@dataclass(frozen=True)
class FlashAttnWheel:
    release: str
    name: str
    url: str
    flash_version: str
    cuda: str
    torch: str
    python_tag: str
    platform_tag: str


def run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    print_command(cmd)
    return subprocess.run(
        cmd,
        cwd=ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def print_command(cmd: list[str]) -> None:
    command = shlex.join(cmd)
    print("実行コマンド: " + command, flush=True)
    with COMMAND_RECORD.open("a", encoding="utf-8") as wf:
        wf.write(command + "\n")


def read_text(path: str) -> str:
    target = ROOT / path
    return target.read_text().strip() if target.exists() else ""


def nvidia_smi() -> dict[str, str | None]:
    if not shutil.which("nvidia-smi"):
        return {"ok": False, "error": "nvidia-smi が見つかりません"}
    plain = run(["nvidia-smi"])
    if plain.returncode != 0:
        return {"ok": False, "error": plain.stderr.strip() or plain.stdout.strip()}
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", plain.stdout)
    result = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    if result.returncode != 0:
        return {"ok": False, "error": result.stderr.strip() or result.stdout.strip()}
    first = result.stdout.strip().splitlines()[0]
    index, name, driver, memory = [part.strip() for part in first.split(",", 3)]
    return {
        "ok": True,
        "index": index,
        "name": name,
        "driver": driver,
        "cuda": cuda_match.group(1) if cuda_match else None,
        "memory": memory,
    }


def latest_pypi_version(package: str) -> str:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/json", timeout=15) as response:
        data = json.load(response)
    return data["info"]["version"]


def flash_attention2_releases() -> list[dict[str, object]]:
    request = urllib.request.Request(
        FLASH_ATTN_RELEASES_URL,
        headers={"Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def parse_flash_attn_wheel(release: str, asset: dict[str, object]) -> FlashAttnWheel | None:
    name = str(asset.get("name", ""))
    match = re.fullmatch(
        r"flash_attn-(?P<flash>[^+]+)\+"
        r"(?P<cuda>cu\d+)"
        r"torch(?P<torch>\d+\.\d+(?:\.\d+)?)"
        r"cxx11abi(?:TRUE|FALSE)-"
        r"(?P<python>cp\d+)-cp\d+-"
        r"(?P<platform>.+)\.whl",
        name,
    )
    if not match:
        return None
    return FlashAttnWheel(
        release=release,
        name=name,
        url=str(asset["browser_download_url"]),
        flash_version=match.group("flash"),
        cuda=match.group("cuda"),
        torch=match.group("torch"),
        python_tag=match.group("python"),
        platform_tag=match.group("platform"),
    )


def target_python_tag() -> str:
    configured = read_text(".python-version")
    match = re.match(r"^(\d+)\.(\d+)", configured)
    if match:
        return f"cp{match.group(1)}{match.group(2)}"
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def target_platform_tag() -> str:
    if sys.platform.startswith("linux"):
        machine = platform.machine().lower()
        if machine in {"x86_64", "amd64"}:
            return "linux_x86_64"
        if machine in {"aarch64", "arm64"}:
            return "linux_aarch64"
    raise RuntimeError(f"未対応 platform です: {platform.platform()} / {platform.machine()}")


def cuda_major(gpu: dict[str, str | None]) -> str:
    cuda = gpu.get("cuda")
    if cuda:
        return "cu" + cuda.split(".", 1)[0]
    return "cu12"


def choose_flash_attn_wheel(gpu: dict[str, str | None]) -> FlashAttnWheel:
    python_tag = target_python_tag()
    platform_tag = target_platform_tag()
    cuda = cuda_major(gpu)
    releases = flash_attention2_releases()

    candidates: list[FlashAttnWheel] = []
    for release in releases:
        tag = str(release.get("tag_name", ""))
        if not tag.startswith("v2."):
            continue
        for asset in release.get("assets", []):
            wheel = parse_flash_attn_wheel(tag, asset)
            if (
                wheel
                and wheel.python_tag == python_tag
                and wheel.platform_tag == platform_tag
                and wheel.cuda == cuda
            ):
                candidates.append(wheel)
        if candidates:
            return max(candidates, key=lambda wheel: tuple(int(x) for x in re.findall(r"\d+", wheel.torch)))

    raise RuntimeError(
        "対応する FlashAttention 2 wheel が見つかりません: "
        f"python={python_tag}, platform={platform_tag}, cuda={cuda}"
    )


def torch_source_name(torch_cuda_index: str) -> str:
    return f"torch-{torch_cuda_index}"


def torch_index_from_gpu(gpu: dict[str, str | None]) -> str:
    cuda = gpu.get("cuda")
    if not cuda:
        return "cu128"
    major_minor = tuple(int(part) for part in cuda.split(".")[:2])
    if major_minor >= (13, 0):
        return "cu130"
    return "cu128"


def torch_pin_version(torch_version: str) -> str:
    parts = torch_version.split(".")
    if len(parts) == 2:
        return torch_version + ".0"
    return torch_version


def replace_section(text: str, header: str, body: str) -> str:
    pattern = rf"(?ms)^{re.escape(header)}\n.*?(?=^\[|\Z)"
    replacement = header + "\n" + body.rstrip() + "\n\n"
    if not re.search(pattern, text):
        raise RuntimeError(f"{header} section が見つかりません")
    return re.sub(pattern, replacement, text, count=1)


def update_pyproject(wheel: FlashAttnWheel, torch_cuda_index: str) -> None:
    torch_source = torch_source_name(torch_cuda_index)
    torch_version = torch_pin_version(wheel.torch)
    text = PYPROJECT.read_text()
    dependencies = (
        "dependencies = [\n"
        f'    "torch=={torch_version}",\n'
        '    "transformers==5.9.0",\n'
        '    "trl==1.5.1",\n'
        f'    "flash-attn @ {wheel.url}",\n'
        '    "ninja",\n'
        '    "packaging",\n'
        '    "psutil",\n'
        "]"
    )
    text = re.sub(r"(?ms)^dependencies = \[(?:\n.*?\n)?\]", dependencies, text, count=1)
    text = re.sub(
        r"(?ms)\n\[project\.optional-dependencies\]\n.*?(?=^\[|\Z)",
        (
            "\n"
        ),
        text,
        count=1,
    )
    sources_body = f'torch = {{ index = "{torch_source}" }}'
    text = replace_section(text, "[tool.uv.sources]", sources_body)
    index_body = (
        f'name = "{torch_source}"\n'
        f'url = "https://download.pytorch.org/whl/{torch_cuda_index}"\n'
        "explicit = true"
    )
    text = replace_section(text, "[[tool.uv.index]]", index_body)
    PYPROJECT.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--torch-cuda-index",
        choices=["auto", "cu128", "cu130"],
        default="auto",
        help="PyTorch index の CUDA variant。auto は NVIDIA driver CUDA から選択。",
    )
    args = parser.parse_args()

    COMMAND_RECORD.write_text("", encoding="utf-8")
    print(f"コマンド記録: {COMMAND_RECORD}", flush=True)
    gpu = nvidia_smi()
    print(f".python-version: {read_text('.python-version') or '(なし)'}", flush=True)
    print(f"Python: {sys.version.split()[0]}", flush=True)
    print(f"Platform: {platform.platform()} / {platform.machine()}", flush=True)
    print(f"uv: {(run(['uv', '--version']).stdout.strip() or '(unknown)')}", flush=True)
    print(f"PyPI torch: {latest_pypi_version('torch')}", flush=True)
    print(f"PyPI transformers: {latest_pypi_version('transformers')}", flush=True)

    if gpu.get("ok"):
        print(
            "GPU: "
            f"{gpu['name']} / driver {gpu['driver']} / driver CUDA {gpu['cuda']} / {gpu['memory']}",
            flush=True,
        )
    else:
        print(f"GPU: 利用不可 ({gpu['error']})")
        print("NVIDIA ドライバが見えないため、同期は行いません。")
        return 1

    wheel = choose_flash_attn_wheel(gpu)
    torch_cuda_index = args.torch_cuda_index
    if torch_cuda_index == "auto":
        torch_cuda_index = torch_index_from_gpu(gpu)
    print(f"FlashAttention 2 release: {wheel.release}", flush=True)
    print(f"FlashAttention 2 wheel: {wheel.name}", flush=True)
    print(f"FlashAttention 2 wheel が要求する torch: {wheel.torch}", flush=True)
    print(f"PyTorch index: {torch_cuda_index}", flush=True)

    if args.dry_run:
        print("dry-run: pyproject.toml は更新せず、uv sync も実行しません。", flush=True)
    else:
        update_pyproject(wheel, torch_cuda_index)
        print("pyproject.toml を選択した wheel / torch に更新しました。", flush=True)

    cmd = ["uv", "sync"]
    if args.dry_run:
        cmd.append("--dry-run")

    print_command(cmd)
    if args.dry_run:
        return 0
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
