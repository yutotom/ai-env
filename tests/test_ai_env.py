from __future__ import annotations

import unittest

from scripts.ai_env import (
    cuda_from_output,
    parse_flash_attn_wheel,
    platform_tag,
    required_cuda,
    torch_index_for_cuda,
    version_tuple,
)


class VersionTests(unittest.TestCase):
    def test_version_tuple_compares_numerically(self) -> None:
        self.assertGreater(version_tuple("13.0"), version_tuple("12.8"))

    def test_cuda_from_nvidia_smi_output(self) -> None:
        output = "NVIDIA-SMI 580.0  Driver Version: 580.0  CUDA Version: 13.0"
        self.assertEqual(cuda_from_output(output), "13.0")
        self.assertIsNone(cuda_from_output("CUDA unavailable"))

    def test_torch_index_selection(self) -> None:
        cases = {"12.6": "cu126", "12.8": "cu128", "13.0": "cu130", "13.1": "cu130"}
        for cuda, expected in cases.items():
            with self.subTest(cuda=cuda):
                self.assertEqual(torch_index_for_cuda(cuda), expected)
        with self.assertRaises(RuntimeError):
            torch_index_for_cuda("12.5")

    def test_required_cuda(self) -> None:
        self.assertEqual(required_cuda("cu126"), (12, 6))
        self.assertEqual(required_cuda("cu130"), (13, 0))


class WheelTests(unittest.TestCase):
    def test_parse_flash_attention_wheel(self) -> None:
        asset = {
            "name": (
                "flash_attn-2.8.3+cu13torch2.10"
                "cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
            ),
            "browser_download_url": "https://example.test/flash-attn.whl",
        }
        wheel = parse_flash_attn_wheel("v2.8.3", asset)
        self.assertIsNotNone(wheel)
        assert wheel is not None
        self.assertEqual(wheel.flash_version, "2.8.3")
        self.assertEqual(wheel.torch, "2.10")
        self.assertEqual(wheel.cuda, "cu13")
        self.assertEqual(wheel.python_tag, "cp312")
        self.assertEqual(wheel.platform_tag, "linux_x86_64")

    def test_reject_non_wheel_asset(self) -> None:
        self.assertIsNone(
            parse_flash_attn_wheel(
                "v2.8.3",
                {"name": "source.tar.gz", "browser_download_url": "https://example.test"},
            )
        )

    def test_platform_aliases(self) -> None:
        self.assertEqual(platform_tag("amd64"), "linux_x86_64")
        self.assertEqual(platform_tag("arm64"), "linux_aarch64")


if __name__ == "__main__":
    unittest.main()
