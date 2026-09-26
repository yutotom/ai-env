#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["tomlkit>=0.13,<1", "packaging>=24"]
# ///
"""マシンに合うAI環境を構成する。"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
import json
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import tomlkit
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from tomlkit.items import Array
from tomlkit.toml_document import TOMLDocument


# インストール先ではなく、ai-envを実行したプロジェクトを更新対象にする。
ROOT = Path.cwd().resolve()
PYPROJECT = ROOT / "pyproject.toml"
PYTHON_VERSION_FILE = ROOT / ".python-version"
FLASH_ATTN_RELEASES_URL = "https://api.github.com/repos/Dao-AILab/flash-attention/releases"
TORCH_INDEXES = ("cu126", "cu128", "cu130")
DEFAULT_PYTHON_VERSION = "3.12.11"


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
    cxx11abi: bool


def run(
    command: list[str], *, capture: bool = False, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """コマンドを標準出力に表示して実行する。"""

    rendered = shlex.join(command)
    print("実行コマンド: " + rendered, flush=True)
    return subprocess.run(
        command,
        cwd=ROOT if cwd is None else cwd,
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


def project_name(root: Path) -> str:
    """ディレクトリ名をuvで扱える安全なプロジェクト名へ変換する。"""

    normalized = re.sub(r"[^a-z0-9]+", "-", root.name.lower()).strip("-")
    return normalized or "ai-env-project"


def ensure_pyproject(root: Path) -> bool:
    """未作成のpyproject.tomlをAI環境更新用の最小構成で生成する。"""

    pyproject = root / "pyproject.toml"
    if pyproject.exists() or pyproject.is_symlink():
        require(
            pyproject.is_file(),
            f"pyproject.toml がファイルではありません: {pyproject}",
        )
        return False

    # READMEなどの追加ファイルを要求せず、空のディレクトリでも初期化できる構成にする。
    pyproject.write_text(initial_pyproject(root), encoding="utf-8")
    return True


def ensure_python_version(root: Path) -> bool:
    """未作成の.python-versionを対応するPython固定バージョンで生成する。"""

    version_file = root / ".python-version"
    if version_file.exists() or version_file.is_symlink():
        require(
            version_file.is_file(),
            f".python-version がファイルではありません: {version_file}",
        )
        return False

    version_file.write_text(DEFAULT_PYTHON_VERSION + "\n", encoding="utf-8")
    return True


def validate_project_root(root: Path) -> None:
    """カレントディレクトリが更新可能な対象プロジェクトか確認する。"""

    missing = [
        name
        for name in ("pyproject.toml", ".python-version")
        if not (root / name).is_file()
    ]
    require(
        not missing,
        "カレントディレクトリに必要なファイルがありません: "
        + ", ".join(missing)
        + f"（対象: {root}）",
    )


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
    for index in sorted(TORCH_INDEXES, key=required_cuda, reverse=True):
        if driver_cuda >= required_cuda(index):
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
        r"cxx11abi(?P<abi>TRUE|FALSE)-(?P<python>cp\d+)-cp\d+-"
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
        match.group("abi") == "TRUE",
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


def choose_flash_attn_wheel(
    torch_index: str, configured_python: str | None = None
) -> FlashAttnWheel:
    """現在のPython・OS・CUDAに一致するFlashAttention 2 wheelを選ぶ。"""

    if configured_python is None:
        configured_python = PYTHON_VERSION_FILE.read_text(encoding="utf-8").strip()
    python_match = re.match(r"^(\d+)\.(\d+)", configured_python)
    python_tag = (
        f"cp{python_match.group(1)}{python_match.group(2)}"
        if python_match
        else f"cp{sys.version_info.major}{sys.version_info.minor}"
    )
    wheels = (
        wheel
        for release in flash_attention2_releases()
        if str(release.get("tag_name", "")).startswith("v2.")
        for asset in release.get("assets", [])
        if (wheel := parse_flash_attn_wheel(str(release["tag_name"]), asset))
    )
    return select_flash_attn_wheel(wheels, torch_index, python_tag, platform_tag())


def select_flash_attn_wheel(
    wheels: Iterable[FlashAttnWheel],
    torch_index: str,
    python_tag: str,
    target_platform: str,
) -> FlashAttnWheel:
    """外部環境を参照せず、互換性と優先順位だけでwheelを選択する。"""

    # FlashAttention側はcu126/cu128を分けず、CUDA majorだけをwheel名に持つ。
    target_cuda = "cu13" if torch_index == "cu130" else "cu12"
    candidates = [
        wheel
        for wheel in wheels
        if wheel.python_tag == python_tag
        and wheel.platform_tag == target_platform
        and wheel.cuda == target_cuda
        # 対応する公式Linux CUDA 12.6以降のwheelはCXX11 ABIを使用する。
        # https://pytorch.org/blog/pytorch2-6/
        and version_tuple(wheel.torch) >= (2, 6)
        and wheel.cxx11abi
    ]
    if not candidates:
        raise RuntimeError(
            "対応する FlashAttention 2 wheel が見つかりません: "
            f"python={python_tag}, platform={target_platform}, cuda={target_cuda}, "
            "cxx11abi=TRUE"
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
    summary = run(["nvidia-smi"], capture=True)
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


def render_pyproject(
    text: str, wheel: FlashAttnWheel, torch_index: str, transformers: str, trl: str
) -> str:
    """管理対象だけを更新し、ユーザーの依存関係・extras・配布元を保持する。"""

    document = tomlkit.parse(text)
    require("project" in document, "[project] がありません。プロジェクト設定を作成してください。")
    project = document["project"]
    require(
        "dependencies" not in project.get("dynamic", []),
        "動的な dependencies は更新できません。静的な依存関係を指定してください。",
    )
    torch_version = wheel.torch if wheel.torch.count(".") == 2 else wheel.torch + ".0"
    replacements = {
        "torch": f"torch=={torch_version}",
        "transformers": f"transformers=={transformers}",
        "trl": f"trl=={trl}",
        "flash-attn": f"flash-attn @ {wheel.url}",
    }
    dependencies = project.setdefault("dependencies", tomlkit.array())
    update_dependencies(dependencies, replacements)
    update_torch_sources(document, torch_index, set(replacements))
    return tomlkit.dumps(document)


def update_dependencies(dependencies: Array, replacements: dict[str, str]) -> None:
    """対象依存だけを置換し、指定位置と対象外のバージョン制約を保持する。"""

    # 最初の指定位置を維持し、対象の重複だけを取り除く。
    seen: set[str] = set()
    duplicates: list[int] = []
    for position in range(len(dependencies)):
        name = canonicalize_name(Requirement(str(dependencies[position])).name)
        if name in replacements:
            if name in seen:
                duplicates.append(position)
            else:
                dependencies[position] = replacements[name]
                seen.add(name)
    for position in reversed(duplicates):
        del dependencies[position]
    for name, dependency in replacements.items():
        if name not in seen:
            dependencies.append(dependency)
    names = {canonicalize_name(Requirement(str(dep)).name) for dep in dependencies}
    for name in ("ninja", "packaging", "psutil"):
        if name not in names:
            dependencies.append(name)


def update_torch_sources(
    document: TOMLDocument, torch_index: str, managed_packages: set[str]
) -> None:
    """対象の古い配布元を除去し、選択したPyTorch indexを設定する。"""

    uv = document.setdefault("tool", {}).setdefault("uv", {})
    sources = uv.setdefault("sources", {})
    # 管理対象の古いsourceが新しい依存指定を上書きしないようにする。
    for name in list(sources):
        if canonicalize_name(name) in managed_packages:
            del sources[name]
    index_name = f"torch-{torch_index}"
    sources["torch"] = {"index": index_name}
    indexes = uv.setdefault("index", tomlkit.aot())
    selected = next((entry for entry in indexes if entry.get("name") == index_name), None)
    if selected is None:
        indexes.append(tomlkit.table())
        selected = indexes[-1]
    selected["name"] = index_name
    selected["url"] = f"https://download.pytorch.org/whl/{torch_index}"
    selected["explicit"] = True


def initial_pyproject(root: Path) -> str:
    """ファイルを作らず、新規プロジェクトの設定を生成する。"""

    return (
        f'[project]\nname = "{project_name(root)}"\nversion = "0.1.0"\n'
        'requires-python = ">=3.12,<3.13"\ndependencies = []\n'
    )


def check_proposed_project(text: str, python_version: str) -> int:
    """一時コピーで更新後の構成を解決し、対象プロジェクトへの書き込みを避ける。"""

    document = tomlkit.parse(text)
    uv = document.get("tool", {}).get("uv", {})
    require(
        "workspace" not in uv,
        "workspace の dry-run は未対応です。単独プロジェクトで確認してください。",
    )
    # 相対pathの参照先を元のプロジェクトと一致させる。
    for source in uv.get("sources", {}).values():
        for entry in source if isinstance(source, list) else [source]:
            if "path" in entry:
                entry["path"] = str((ROOT / entry["path"]).resolve())
    with tempfile.TemporaryDirectory(prefix="ai-env-check-") as directory:
        temporary_root = Path(directory)
        # ビルド設定やREADMEもコピーし、元のファイルに書き込まない。
        shutil.copytree(
            ROOT, temporary_root, dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".venv", ".git", "__pycache__", ".pytest_cache"),
        )
        (temporary_root / "pyproject.toml").write_text(tomlkit.dumps(document), encoding="utf-8")
        (temporary_root / ".python-version").write_text(python_version + "\n", encoding="utf-8")
        return run(
            ["uv", "sync", "--dry-run", "--python", python_version],
            cwd=temporary_root,
        ).returncode


def print_configuration(
    python_version: str,
    gpu: dict[str, str],
    packages: dict[str, str],
    wheel: FlashAttnWheel,
    torch_index: str,
) -> None:
    """調査した環境と選択した構成を利用者へ表示する。"""

    print(f".python-version: {python_version}", flush=True)
    print(f"Python: {sys.version.split()[0]}", flush=True)
    print(f"Platform: {platform.platform()} / {platform.machine()}", flush=True)
    uv = run(["uv", "--version"], capture=True)
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


def configure(args: argparse.Namespace) -> int:
    """ホストに合う依存バージョンを決定し、uvで環境を同期する。"""

    # メモリ上で補完し、GPU調査やdry-runの失敗時にファイルを残さない。
    text = (
        PYPROJECT.read_text(encoding="utf-8")
        if PYPROJECT.exists() else initial_pyproject(ROOT)
    )
    python_version = (
        PYTHON_VERSION_FILE.read_text(encoding="utf-8").strip()
        if PYTHON_VERSION_FILE.exists() else DEFAULT_PYTHON_VERSION
    )
    require(
        re.fullmatch(r"3\.12(?:\.\d+)?", python_version) is not None,
        "Python 3.12 が必要です。.python-version を確認してください。",
    )
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
        wheel = choose_flash_attn_wheel(torch_index, python_version)
    except (OSError, RuntimeError) as error:
        return fail(str(error))

    print_configuration(python_version, gpu, packages, wheel, torch_index)
    proposed = render_pyproject(
        text, wheel, torch_index, packages["transformers"], packages["trl"]
    )
    if args.dry_run:
        print("dry-run: 一時コピーで選択した構成を検証します。対象ファイルは変更しません。")
        return check_proposed_project(proposed, python_version)
    PYPROJECT.write_text(proposed, encoding="utf-8")
    ensure_python_version(ROOT)
    print("pyproject.toml を選択した互換バージョンに更新しました。", flush=True)
    return run(["uv", "sync"]).returncode


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
        return fail(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
