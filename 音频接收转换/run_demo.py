"""
AudioPipeline 本地演示 — 采集 5 秒麦克风音频并保存。

运行:
    python run_demo.py
    python run_demo.py --list    (列出所有麦克风)
    python run_demo.py --device 2  (选择指定设备)
"""

from __future__ import annotations

import argparse
import time
import numpy as np
import soundfile as sf
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="AudioPipeline 演示")
    parser.add_argument("--list", action="store_true", help="列出可用麦克风")
    parser.add_argument("--device", type=int, default=None, help="设备索引")
    parser.add_argument("--duration", type=int, default=5, help="采集时长 (秒)")
    parser.add_argument("--output", type=str, default="demo_output.wav")
    args = parser.parse_args()

    from audio_receiver import AudioPipeline

    # 列出设备
    devices = AudioPipeline.get_device_list()
    if args.list or not devices:
        print("=== 可用麦克风 ===")
        for d in devices:
            print(f"  [{d.index}] {d.name}  ({d.channels}ch, {d.sample_rate}Hz)")
        if not devices:
            print("  ⚠️ 未检测到麦克风设备")
        if args.list:
            return
        if not devices:
            print("\n无麦克风可用，切换到合成音频演示模式...")
            run_synthetic_demo(args.output)
            return

    # 启动采集
    pipeline = AudioPipeline(
        device_index=args.device,
        chunk_duration_ms=100,
        signal_debug=True,
    )

    print(f"\n🎤 开始采集 {args.duration} 秒...")
    pipeline.start()

    chunks = []
    n_frames = int(args.duration * 1000 / pipeline.chunk_duration_ms)

    for i in range(n_frames):
        chunk = pipeline.read_chunk()
        if chunk is not None:
            chunks.append(chunk)
            if (i + 1) % 10 == 0:
                print(f"  已采集: {(i + 1) * pipeline.chunk_duration_ms}ms")

    pipeline.stop()

    if chunks:
        audio = np.concatenate(chunks, axis=1)
        sf.write(args.output, audio.T, pipeline.TARGET_SR)
        print(f"\n✅ 已保存到: {args.output}")
        print(f"   形状: {audio.shape}, 时长: {audio.shape[1]/16000:.1f}s")
        print(f"   RMS: {np.sqrt(np.mean(audio**2)):.3f}")
    else:
        print("\n⚠️ 未采集到有效音频（可能麦克风静音或权限不足）")


def run_synthetic_demo(output: str):
    """无麦克风时的合成演示 — 生成测试音频模拟 Pipeline 输出."""
    print("\n📡 合成音频演示模式")
    print("   生成 3 秒测试音频 (440Hz + 白噪声)...")

    from audio_receiver import AudioPipeline

    pipeline = AudioPipeline(signal_debug=True)

    # 生成合成音频并走一遍处理流水线
    t = np.linspace(0, 3, 48000, endpoint=False)
    sine = 0.3 * np.sin(2 * np.pi * 440 * t)
    noise = 0.02 * np.random.randn(48000)
    raw = (sine + noise).astype(np.float32).reshape(1, -1)

    # 重采样
    resampled = pipeline._resample(raw, 48000, 16000)

    # AGC
    agc = pipeline._apply_agc(resampled)

    # 伪空间合成
    spatial = pipeline._convert_channels(agc)

    # 标准化
    standardized = pipeline._standardize(spatial)

    sf.write(output, standardized.T, 16000)
    print(f"\n✅ 已保存到: {output}")
    print(f"   形状: {standardized.shape}, 时长: {standardized.shape[1]/16000:.1f}s")
    print("   可送入 SELD 引擎进行推理")


if __name__ == "__main__":
    main()
