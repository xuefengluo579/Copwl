#!/usr/bin/env python3
"""下载 w2v-SELD 预训练模型权重."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Optional

# ── 已知权重 URL ──────────────────────────────────────────

CHECKPOINTS = {
    "base": {
        "name": "w2v_seld_base.pt",
        "url": "https://drive.google.com/uc?id=1PWZH6OpbPlUOZvOgRrMb46z-r89bY1u2",
        "size_mb": 400,
        "description": "BASE 配置 — 快速推理，资源占用低（约 4GB 显存）",
    },
    "large": {
        "name": "w2v_seld_large.pt",
        "url": "https://drive.google.com/uc?id=LARGE_PLACEHOLDER",
        "size_mb": 1200,
        "description": "LARGE 配置 — 更高精度，适合微调（约 8GB 显存）",
    },
}


def download_with_gdown(url: str, output_path: Path) -> bool:
    """使用 gdown 下载 Google Drive 文件."""
    try:
        import gdown
    except ImportError:
        print("[!] gdown not installed. Run: pip install gdown")
        return False

    print(f"[→] 下载: {output_path.name}")
    try:
        gdown.download(url, str(output_path), quiet=False)
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as e:
        print(f"[✗] gdown 下载失败: {e}")
        return False


def download_with_requests(url: str, output_path: Path) -> bool:
    """使用 requests 下载（备用方案，带进度条）."""
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        print("[!] requests/tqdm not installed.")
        return False

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024 * 1024  # 1 MB

        with open(output_path, "wb") as f:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=output_path.name,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return output_path.exists()
    except Exception as e:
        print(f"[✗] requests 下载失败: {e}")
        return False


def verify_file(path: Path, expected_md5: Optional[str] = None) -> bool:
    """验证文件完整性."""
    if not path.exists():
        return False

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"[✓] 文件大小: {size_mb:.1f} MB")

    if size_mb < 1:
        print("[✗] 文件异常小，可能下载失败")
        return False

    if expected_md5:
        print("[→] 验证 MD5...")
        md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        actual = md5.hexdigest()
        if actual != expected_md5:
            print(f"[✗] MD5 不匹配: {actual} != {expected_md5}")
            return False
        print("[✓] MD5 校验通过")

    return True


def main():
    parser = argparse.ArgumentParser(description="下载 w2v-SELD 预训练权重")
    parser.add_argument(
        "--variant",
        choices=["base", "large", "all"],
        default="base",
        help="要下载的模型变体 (default: base)",
    )
    parser.add_argument(
        "--output-dir",
        default="./checkpoints",
        help="输出目录 (default: ./checkpoints)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载（即使文件已存在）",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = ["base", "large"] if args.variant == "all" else [args.variant]

    success = True
    for variant in variants:
        info = CHECKPOINTS[variant]
        output_path = output_dir / info["name"]

        print(f"\n{'=' * 50}")
        print(f" {variant.upper()} — {info['description']}")
        print(f" 预计大小: ~{info['size_mb']} MB")
        print(f"{'=' * 50}")

        if output_path.exists() and not args.force:
            if verify_file(output_path):
                print(f"[✓] {info['name']} 已存在，跳过 (--force 可强制重新下载)")
                continue
            else:
                print(f"[!] {info['name']} 文件损坏，重新下载...")

        # 尝试 gdown → requests fallback
        downloaded = download_with_gdown(info["url"], output_path)
        if not downloaded:
            print("[!] gdown 失败，尝试 requests...")
            downloaded = download_with_requests(info["url"], output_path)

        if not downloaded:
            print(f"[✗] {variant} 下载失败")
            success = False
            continue

        # 验证
        if not verify_file(output_path):
            print(f"[✗] {variant} 验证失败")
            success = False
        else:
            print(f"[✓] {variant} 下载完成: {output_path}")

    if success:
        print(f"\n[✓] 所有权重下载完成！")
        print(f"   放置位置: {output_dir.resolve()}")
        print(f"   配置文件已指向此目录，可直接使用。")
        sys.exit(0)
    else:
        print(f"\n[✗] 部分权重下载失败，请检查网络连接。")
        print(f"   手动下载: https://github.com/Orlllem/seld_wav2vec2")
        sys.exit(1)


if __name__ == "__main__":
    main()
