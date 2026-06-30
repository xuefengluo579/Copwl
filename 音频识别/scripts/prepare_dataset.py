#!/usr/bin/env python3
"""
训练数据准备工具。

功能:
  1. 从标注音频切分训练样本
  2. 生成 DCASE SELD 格式的标注 CSV
  3. 单通道 → 四通道伪 Ambisonics 转换（数据增强）
  4. 生成 train/val/test 划分

输入: 一个包含 .wav 文件和 annotations.csv 的文件夹
输出: DCASE 标准格式的 dataset/ 目录

标注文件格式 (annotations.csv):
    filename,onset,offset,event_class
    scene_001.wav,0.5,1.2,fall
    scene_001.wav,3.0,3.8,knock
    scene_002.wav,0.0,0.8,glass_break
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import soundfile as sf
from scipy import signal


# ── 事件类别映射 ──────────────────────────────────────────

EVENT_CLASSES = [
    "speech",
    "fall",
    "glass_break",
    "knock",
    "object_drop",
    "footstep",
    "door_slam",
]


def load_annotations(csv_path: Path) -> List[Tuple[str, float, float, str]]:
    """
    加载标注文件.

    Returns:
        [(filename, onset_sec, offset_sec, event_class), ...]
    """
    annotations = []
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 4:
                filename = parts[0].strip()
                onset = float(parts[1])
                offset = float(parts[2])
                event_class = parts[3].strip()
                if event_class in EVENT_CLASSES:
                    annotations.append((filename, onset, offset, event_class))
    return annotations


def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    target_sr: int = 16000,
    target_channels: int = 4,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    use_augmentation: bool = True,
    rir_dir: Optional[Path] = None,
):
    """
    准备 DCASE SELD 格式的训练数据集.

    Args:
        input_dir: 输入目录，包含 .wav 文件和 annotations.csv
        output_dir: 输出目录
        target_sr: 目标采样率
        target_channels: 目标通道数 (4 = Ambisonics)
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        use_augmentation: 是否做数据增强 (伪空间合成 + 随机延迟)
        rir_dir: 空间脉冲响应目录 (可选, 用于真实 Ambisonics 合成)
    """
    csv_path = input_dir / "annotations.csv"
    if not csv_path.exists():
        print(f"❌ 未找到标注文件: {csv_path}")
        print(f"   请在 {input_dir} 下创建 annotations.csv")
        return

    annotations = load_annotations(csv_path)
    if not annotations:
        print("❌ 标注文件为空或格式错误")
        return

    print(f"📋 加载了 {len(annotations)} 条标注")

    # 收集所有涉及的音频文件
    audio_files = sorted(set(fn for fn, _, _, _ in annotations))

    # 生成 train/val/test 划分 (按文件划分, 避免同文件跨集合)
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(audio_files))
    n_test = max(1, int(len(audio_files) * test_ratio))
    n_val = max(1, int(len(audio_files) * val_ratio))
    n_train = len(audio_files) - n_val - n_test

    splits = {}
    for i, split_name, count in [
        (indices[:n_train], "train", n_train),
        (indices[n_train : n_train + n_val], "val", n_val),
        (indices[n_train + n_val :], "test", n_test),
    ]:
        for idx in i:
            splits[audio_files[idx]] = split_name

    # 创建输出目录
    for subset in ["train", "val", "test"]:
        (output_dir / "audio").mkdir(parents=True, exist_ok=True)
        metadata_dir = output_dir / "metadata" / subset
        metadata_dir.mkdir(parents=True, exist_ok=True)

    # 按划分收集每个文件的标注
    subset_annotations = {s: {} for s in ["train", "val", "test"]}
    for filename, onset, offset, event_class in annotations:
        subset = splits.get(filename, "train")
        if filename not in subset_annotations[subset]:
            subset_annotations[subset][filename] = []
        subset_annotations[subset][filename].append((onset, offset, event_class))

    # 处理每个音频文件
    for subset_name in ["train", "val", "test"]:
        subset_files = subset_annotations[subset_name]
        if not subset_files:
            continue

        print(f"\n📂 {subset_name}: {len(subset_files)} 个文件")

        for filename, events in subset_files.items():
            src_path = input_dir / filename
            if not src_path.exists():
                print(f"  ⚠️ 跳过不存在的文件: {filename}")
                continue

            # 加载音频
            audio, sr = load_audio_raw(src_path, target_sr, target_channels)
            if audio is None:
                continue

            base_name = Path(filename).stem
            dst_audio = output_dir / "audio" / f"{base_name}.wav"
            sf.write(str(dst_audio), audio.T, target_sr)

            # 写入标注 CSV
            csv_out = output_dir / "metadata" / subset_name / f"{base_name}.csv"
            with open(csv_out, "w", encoding="utf-8") as f:
                f.write("onset,offset,event_class,doa_x,doa_y,doa_z\n")
                for onset, offset, event_class in events:
                    f.write(f"{onset:.2f},{offset:.2f},{event_class},0,0,0\n")

            # 数据增强 (仅训练集)
            if subset_name == "train" and use_augmentation:
                augment_audio(
                    audio, base_name, events, output_dir, target_sr
                )

    # 统计
    print("\n" + "=" * 50)
    print("✅ 数据集准备完成!")
    for s in ["train", "val", "test"]:
        n_files = len(subset_annotations[s])
        n_events = sum(len(v) for v in subset_annotations[s].values())
        print(f"  {s:5s}: {n_files:3d} 个文件, {n_events:3d} 个事件")
    print(f"  输出: {output_dir.resolve()}")
    print("=" * 50)


def load_audio_raw(
    path: Path,
    target_sr: int,
    target_channels: int,
) -> Optional[np.ndarray]:
    """加载音频文件并转换为目标格式."""
    try:
        audio, sr = sf.read(str(path), dtype="float32")
    except Exception as e:
        print(f"  ❌ 无法加载 {path}: {e}")
        return None

    # 转置: sf 输出 (samples, channels) → (channels, samples)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    else:
        audio = audio.T

    # 重采样
    if sr != target_sr:
        new_len = int(audio.shape[1] * target_sr / sr)
        resampled = np.zeros((audio.shape[0], new_len), dtype=np.float32)
        for ch in range(audio.shape[0]):
            resampled[ch] = signal.resample(audio[ch], new_len)
        audio = resampled

    # 通道处理
    if audio.shape[0] == 1 and target_channels == 4:
        audio = mono_to_ambisonics(audio[0])
    elif audio.shape[0] != target_channels:
        out = np.zeros((target_channels, audio.shape[1]), dtype=np.float32)
        ch = min(audio.shape[0], target_channels)
        out[:ch] = audio[:ch]
        audio = out

    return audio


def mono_to_ambisonics(mono: np.ndarray) -> np.ndarray:
    """单通道 → 四通道伪 Ambisonics."""
    samples = len(mono)
    out = np.zeros((4, samples), dtype=np.float32)
    out[0] = mono  # W
    delay = int(0.5e-3 * 16000)
    out[1, delay:] = mono[:-delay] * 0.8 if delay < samples else mono * 0.8  # X
    out[2] = mono * 0.6  # Y
    out[3] = mono * 0.5  # Z
    return out


def augment_audio(
    audio: np.ndarray,
    base_name: str,
    events: List[Tuple[float, float, str]],
    output_dir: Path,
    target_sr: int,
):
    """数据增强: 随机延迟 ±20ms + 音量微调."""
    rng = np.random.RandomState(hash(base_name) % 2**32)

    delay_samples = rng.randint(-320, 320)  # -20ms ~ +20ms
    gain = 10 ** (rng.uniform(-2, 2) / 20)  # -2dB ~ +2dB

    augmented = np.roll(audio, delay_samples, axis=1) * gain

    aug_name = f"{base_name}_aug"
    dst = output_dir / "audio" / f"{aug_name}.wav"
    sf.write(str(dst), augmented.T, target_sr)

    csv_out = output_dir / "metadata" / "train" / f"{aug_name}.csv"
    with open(csv_out, "w", encoding="utf-8") as f:
        f.write("onset,offset,event_class,doa_x,doa_y,doa_z\n")
        delay_sec = delay_samples / target_sr
        for onset, offset, event_class in events:
            f.write(
                f"{onset + delay_sec:.2f},{offset + delay_sec:.2f},"
                f"{event_class},0,0,0\n"
            )


# ── CLI ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="准备 DCASE SELD 格式训练数据集"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="输入目录 (包含 .wav 和 annotations.csv)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("./dataset"),
        help="输出目录 (default: ./dataset)",
    )
    parser.add_argument(
        "--sr", type=int, default=16000,
        help="目标采样率 (default: 16000)",
    )
    parser.add_argument(
        "--channels", type=int, default=4,
        help="目标通道数 (default: 4)",
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="禁用数据增强",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="验证集比例 (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.15,
        help="测试集比例 (default: 0.15)",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"❌ 输入目录不存在: {args.input_dir}")
        return

    prepare_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_sr=args.sr,
        target_channels=args.channels,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        use_augmentation=not args.no_augment,
    )


if __name__ == "__main__":
    main()
